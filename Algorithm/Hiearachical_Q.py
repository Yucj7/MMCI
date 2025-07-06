import numpy as np
import math
import random
import env.global_config as gc
import env.env_config as ec
class Hierarchical_Q(object):

    def __init__(self):
        self.layer_set = ec.layer_space
        self.type_set = ec.type_space
        self.rssi_set = ec.RssiSet.tolist()
        self.z_set = ec.TaskNum.tolist()
        self.delay_set = ec.LatencySet.tolist()
        self.energy_set = ec.EnergySet.tolist()
        self.state_space_dim = ec.state_dim
        self.master_policy_dim = ec.type_dim
        self.sub_policy_dim = ec.layer_dim
        self.R_max = 10

        self.v_table = np.zeros((self.state_space_dim, self.master_policy_dim)) + self.R_max
        self.q_table = np.zeros((self.state_space_dim * self.master_policy_dim, self.sub_policy_dim)) + self.R_max

        self.E_v_table = np.zeros((self.state_space_dim, self.master_policy_dim)) + self.R_max
        self.E_q_table = np.zeros((self.state_space_dim * self.master_policy_dim, self.sub_policy_dim)) + self.R_max

        self.epsi_high = 1.0
        self.epsi_low = 0.15
        self.decay = 800
        self.is_anneal = 1
        self.learning_begin = 10
        self.alpha = 0.6
        self.gamma = 0.9


    def getMasterPolicy(self, state, step):
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * step / self.decay))

        if np.random.rand() < epsi:
            action = random.randrange(self.master_policy_dim)
        else:
            action = np.argmax(self.v_table[state])
        return action

    # get sub policy
    def getSubPolicy(self, state, step):
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * step / self.decay))
        if np.random.rand() < epsi:
            action = random.randrange(self.sub_policy_dim)
        else:
            action = np.argmax(self.q_table[state])
        return action

    # <s, a, r, s'>
    def updateMasterPolicy(self, state, action, reward, next_state):
        current_v = self.v_table[state][action]
        new_v = reward + self.gamma * max(self.v_table[next_state])
        self.v_table[state][action] += self.alpha * (new_v - current_v)

    # <s, a, r, s'>
    def updateSubPolicy(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        new_q = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (new_q - current_q)

    # agent reset
    def reset(self):
        self.v_table = np.zeros((self.state_space_dim, self.master_policy_dim)) + self.R_max
        self.q_table = np.zeros((self.state_space_dim * self.master_policy_dim, self.sub_policy_dim)) + self.R_max

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


        delay_index = 0
        for index in self.delay_set:
            if state[2] >= index:
                delay_index = self.delay_set.index(index)
            else:
                break
        energy_index = 0
        for index in self.energy_set:
            if state[3] >= index:
                energy_index = self.energy_set.index(index)
            else:
                break
        state_index = [h_index, z_index, delay_index, energy_index]
        s_idx = h_index * len(self.rssi_set) + z_index * len(self.z_set) + \
                delay_index * len(self.delay_set) + energy_index * len(self.energy_set)

        return s_idx






