import numpy as np
import math
import random
import env.global_config as gc

class Q_learning(object):

    def __init__(self):
        self.action_set = gc.layer_space
        self.rssi_set = gc.rssi_set.tolist()
        self.z_set = gc.Task_Num.tolist()
        self.delay_set =gc.Latency.tolist()
        self.energy_set = gc.Energy.tolist()
        self.tran_delay_set = gc.Trans_Latency.tolist()
        self.state_space_dim = gc.state_dim
        self.R_max = 0
        self.q_table = np.zeros((self.state_space_dim ,len(self.action_set))) + self.R_max

        self.max_epsilon = 1.0
        self.min_spsilon = 0.1
        self.decay = 300
        self.is_anneal = 1
        self.learning_begin = 10
        self.alpha = 0.9
        self.gamma = 0.9


    def choose_action(self, state, step):
        eps_temp = self.min_spsilon + (self.max_epsilon - self.min_spsilon) * math.exp(
            -1. * (step - self.learning_begin) / self.decay)
        if np.random.rand() < eps_temp:
            action = random.randrange(len(self.action_set))
        else:
            action = np.argmax(self.q_table[state])
        return action

    def get_state_index(self, state):
        h_index = 0
        rssi = state[0]
        for index in self.rssi_set:
            if rssi >= index:
                h_index = self.rssi_set.index(index)
            else:
                break

        z_index = 0
        for index in self.z_set:
            if state[1] >= index:
                z_index = self.z_set.index(index)
            else:
                break

        tran_delay_index = 0
        for index in self.tran_delay_set:
            if state[2] >= index:
                tran_delay_index = self.tran_delay_set.index(index)
            else:
                break

        delay_index = 0
        for index in self.delay_set:
            if state[3] >= index:
                delay_index = self.delay_set.index(index)
            else:
                break
        energy_index = 0
        for index in self.energy_set:
            if state[4] >= index:
                energy_index = self.energy_set.index(index)
            else:
                break
        state_index = [h_index, z_index, tran_delay_index, delay_index, energy_index]
        s_idx = h_index * len(self.rssi_set) + z_index * len(self.z_set) + tran_delay_index * len(self.tran_delay_set) + \
                delay_index * len(self.delay_set) + energy_index * len(self.energy_set)
        # print("state index:",state_index)
        return s_idx

    def updatePolicy(self, state, action, reward, next_state):
        # print(state,action,next_state)
        current_q = self.q_table[state][action]
        new_q = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (new_q - current_q)

    def reset(self):
        self.q_table = np.zeros((self.state_space_dim ,len(self.action_set))) + self.R_max








