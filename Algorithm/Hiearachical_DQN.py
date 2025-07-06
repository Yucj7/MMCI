# -*- coding: utf-8 -*-
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import env.global_config as gc
import env.env_config as ec
import os

USE_GPU = False

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#dtype = torch.cuda.FloatTensor # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('using device:', device)


TARGET_REPLACE_ITER = 50  # target update frequency 100


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2.weight.data.normal_(0, 0.1)

class HDQNAgent(object):
    def __init__(self, **kwargs):
        self.layer_set = ec.layer_space
        self.type_set = ec.type_space
        self.rssi_set = ec.RssiSet.tolist()
        self.z_set = ec.TaskNum.tolist()
        self.delay_set = ec.LatencySet.tolist()
        self.energy_set = ec.EnergySet.tolist()
        self.state_space_dim = 4
        self.master_policy_dim = ec.type_dim
        self.sub_policy_dim = ec.layer_dim
        self.R_max = 30
        self.joint_state_space_dim = 4 + 1
        self.lr = 0.01
        self.capacity = 20000
        self.batch_size = 64


        self.VQ_eval_net = Net(self.state_space_dim, 64, self.master_policy_dim).to(device)
        self.VQ_target_net = Net(self.state_space_dim, 64, self.master_policy_dim).to(device)  # 256
        self.VQ_optimizer = optim.Adam(self.VQ_eval_net.parameters(), lr=self.lr)


        self.QQ_eval_net = Net(self.joint_state_space_dim, 64, self.sub_policy_dim).to(device)
        self.QQ_target_net = Net(self.joint_state_space_dim, 64, self.sub_policy_dim).to(device)  # .to(device)  # 256
        self.QQ_optimizer = optim.Adam(self.QQ_eval_net.parameters(), lr=self.lr)


        self.buffer = []

        self.learn_step_counter = 0

        self.epsi_high = 1.0
        self.epsi_low = 0.01
        self.decay = 300
        self.is_anneal = 1
        self.learning_begin = 10
        self.alpha = 0.01
        self.gamma = 0.5


    def getMasterPolicy(self, s0, step):
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-2.0 * step / self.decay))
        if random.random() < epsi:
            # a0 = random.choice([30,31,32,33,34,35])
            a0 = random.randrange(self.master_policy_dim)
        else:
            s0 = torch.tensor(s0, dtype=torch.float, device=device).view(1, -1)
            a0 = torch.argmax(self.VQ_eval_net(s0)).item()
        return a0

    def getSubPolicy(self, s0, step):

        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-2.0 * step / self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.sub_policy_dim)
        else:
            s0 = torch.tensor(s0, dtype=torch.float,device=device).view(1, -1)
            a0 = torch.argmax(self.QQ_eval_net(s0)).item()
        return a0


    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def update_V_Net(self):
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1, a, b, c = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float, device=device)
        a0 = torch.tensor(a0, dtype=torch.long,device=device).view(self.batch_size, -1)
        r1 = torch.tensor(r1, dtype=torch.float, device=device).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float, device=device)

        Q_true = r1 + self.gamma * torch.max(self.VQ_target_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        Q_pred = self.VQ_eval_net(s0).gather(1, a0)


        loss_fn = nn.MSELoss()
        Q_loss = loss_fn(Q_pred, Q_true)
        self.VQ_optimizer.zero_grad()
        Q_loss.backward()
        self.VQ_optimizer.step()


        self.learn_step_counter += 1

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.VQ_target_net.load_state_dict(self.VQ_eval_net.state_dict())

    def update_Q_Net(self):
        if (len(self.buffer)) < self.batch_size:
            return
        samples = random.sample(self.buffer, self.batch_size)
        a, b, r1, c, s0, a0, s1= zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float,device=device)
        a0 = torch.tensor(a0, dtype=torch.long,device=device).view(self.batch_size, -1)
        r1 = torch.tensor(r1, dtype=torch.float,device=device).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float,device=device)


        Q_true = r1 + self.gamma * torch.max(self.QQ_target_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        Q_pred = self.QQ_eval_net(s0).gather(1, a0)

        loss_fn = nn.MSELoss()

        Q_loss = loss_fn(Q_pred, Q_true)

        self.QQ_optimizer.zero_grad()
        Q_loss.backward()
        self.QQ_optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:

            self.QQ_target_net.load_state_dict(self.QQ_eval_net.state_dict())

    
    def net_reset(self):
        self.VQ_eval_net.reset_parameters()
        self.VQ_target_net.reset_parameters()


        self.QQ_eval_net.reset_parameters()
        self.QQ_target_net.reset_parameters()


        self.buffer = []
        self.learn_step_counter = 0

    def save_model(self, filename, directory):

        torch.save(self.eval_net.state_dict(), '%s/%s_eval_net.pth' % (directory, filename))

    def load_model(self, filename, directory):

        self.eval_net.load_state_dict(torch.load('%s/%s_eval_net.pth' % (directory, filename)))







