#import gym
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import env.env_config as ec
import pylab
import matplotlib.pyplot as plt
USE_GPU = True

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
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


class SDQNAgent(object):
    def __init__(self, **kwargs):
        self.layer_set = ec.layers_space
        self.com_set = ec.compress_space
        self.rssi_set = ec.rssi_set.tolist()
        self.z_set = ec.Task_Num.tolist()
        self.delay_set = ec.Latency.tolist()
        self.energy_set = ec.Energy.tolist()
        self.state_space_dim = 4
        self.action_space_dim = ec.action_dim
        self.R_max = 30
        self.lr = 0.01
        self.capacity = 20000
        self.batch_size = 64

        self.Q_eval_net = Net(self.state_space_dim, 64, self.action_space_dim).to(device)
        self.Q_target_net = Net(self.state_space_dim, 64, self.action_space_dim).to(device)  # 256
        self.E_eval_net = Net(self.state_space_dim, 64, self.action_space_dim).to(device)
        self.E_target_net = Net(self.state_space_dim, 64, self.action_space_dim).to(device)  # 256
        self.Q_optimizer = optim.Adam(self.Q_eval_net.parameters(), lr=self.lr)
        self.E_optimizer = optim.Adam(self.E_eval_net.parameters(), lr=self.lr)

        self.buffer = []

        self.learn_step_counter = 0

        self.epsi_high = 1.0
        self.epsi_low = 0.1
        self.decay = 1000
        self.is_anneal = 1
        self.learning_begin = 10
        self.alpha = 0.6
        self.gamma = 0.9
        self.beta = 0.2

    def act(self, s0):

        s0 = torch.tensor(s0, dtype=torch.float, device=device).view(1, -1)
        Q_value = self.Q_eval_net(s0)
        E_value = self.E_eval_net(s0)
        Q_offset = 1
        try:
            temp = (math.e ** (Q_offset * Q_value / (E_value+1))).reshape(-1)
            prob = temp / sum(temp)
            prob = prob.cpu().detach().numpy()
            a0 = np.random.choice(range(self.action_space_dim), 1, replace=False, p=prob)[0]
            #a0 = torch.multinomial(prob, self.action_space_dim, replacement=False).item()

        except:
            a0 = np.random.choice(range(self.action_space_dim))#torch.argmax(self.Q_eval_net(s0)).item()
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, e1, s1 = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float,device=device)
        a0 = torch.tensor(a0, dtype=torch.long,device=device).view(self.batch_size, -1)
        r1 = torch.tensor(r1, dtype=torch.float,device=device).view(self.batch_size, -1)
        e1 = torch.tensor(e1, dtype=torch.float, device=device).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float,device=device)

        # ===================================================================#
        y_next_Q = self.Q_eval_net(s1)
        argmax_Q = torch.argmax(y_next_Q, dim=1)
        q_next = self.Q_target_net(s1).detach()

        q_updata = torch.zeros((self.batch_size, 1), dtype=torch.float, device=device)
        for i in range(self.batch_size):
            q_updata[i] = q_next[i, argmax_Q[i]]

        q_updata = self.gamma * q_updata
        Q_true = r1 + q_updata
        # ========================================================#


    #    Q_true = r1 + self.gamma * torch.max(self.Q_target_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        Q_pred = self.Q_eval_net(s0).gather(1, a0)

        E_true = e1 + self.beta * torch.min(self.E_target_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        E_pred = self.E_eval_net(s0).gather(1, a0)

        loss_fn = nn.MSELoss()

        Q_loss = loss_fn(Q_pred, Q_true)
        E_loss = loss_fn(E_pred, E_true)

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        self.E_optimizer.zero_grad()
        E_loss.backward()
        self.E_optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:

            self.Q_target_net.load_state_dict(self.Q_eval_net.state_dict())
            self.E_target_net.load_state_dict(self.E_eval_net.state_dict())

    def net_reset(self):
        self.Q_eval_net.reset_parameters()
        self.Q_target_net.reset_parameters()
        self.E_eval_net.reset_parameters()
        self.E_target_net.reset_parameters()

        self.buffer = []
        self.learn_step_counter = 0
        self.steps = 0

    def save_model(self, filename, directory):

        torch.save(self.eval_net.state_dict(), '%s/%s_eval_net.pth' % (directory, filename))

    def load_model(self, filename, directory):

        self.eval_net.load_state_dict(torch.load('%s/%s_eval_net.pth' % (directory, filename)))
