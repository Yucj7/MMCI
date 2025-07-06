# File :     env_config.py
# Author :   Chuxuan,Changjin
# Project :  Sim-DMMCI

import itertools
import pprint

import scipy.io
import numpy as np

TaskNum = np.arange(1, 6, 1)
RssiSet = np.arange(-60, -38, 2)
LatencySet = np.arange(0.01, 0.6, 0.05)
TranSet = np.arange(0.1, 2, 0.2)
EnergySet = np.arange(0.1, 2, 0.2)

state_dim =len(RssiSet) * len(TaskNum) * len(LatencySet) * len(EnergySet)

M = 6
Type = list(range(2))
Layer = list(range(2))
Node = list(range(1))
BandWidth = [2.2*10**7, 2.4*10**7, 3.0*10**7, 3.6*10**7, 3.8*10**7]

power = 100
running_power = 2
mBase = 0.8

layer_space = list(itertools.product(Layer, repeat=M))
type_space = list(itertools.product(Type, repeat=M))
node_space = list(itertools.product(Node, repeat=M))
type_dim = len(type_space)
layer_dim = len(layer_space)
node_dim = len(node_space)


join_state_size = state_dim * type_dim * node_dim
State_masterPolicy_combination = np.arange(join_state_size).reshape(state_dim, type_dim)


def get_action(index):
    return int(index/layer_dim), index%layer_dim

def get_B(h, step):
    if step > 120:
        BandWidth = [3.0 * 10 ** 7, 3.2 * 10 ** 7, 3.4 * 10 ** 7, 3.6 * 10 ** 7, 3.8 * 10 ** 7]
    else:
        BandWidth = [2.2 * 10 ** 7, 2.4 * 10 ** 7, 3.0 * 10 ** 7, 3.6 * 10 ** 7, 3.8 * 10 ** 7]
    if h < -56:
        return BandWidth[0]
    if h >= -56 and h <-52:
        return BandWidth[1]
    if h >= -52 and h < -46:
        return BandWidth[2]
    if h >= -46 and h <-42:
        return BandWidth[3]
    if h >= -42:
        return BandWidth[4]

def get_b(h):
    if h < -56:
        return BandWidth[0]
    if h >= -56 and h <-52:
        return BandWidth[1]
    if h >= -52 and h < -46:
        return BandWidth[2]
    if h >= -46 and h <-42:
        return BandWidth[3]
    if h >= -42:
        return BandWidth[4]



mat_data = scipy.io.loadmat('results.mat')
results = {}

for i in range(1, 6):
    var_name = f'results_{i}'
    if var_name in mat_data:
        raw_results = mat_data[var_name]
        results_dict = {}

        for key in raw_results.dtype.names:
            items = raw_results[key][0, 0]
            result_list = []

            for entry in items:
                time_cost = float(entry[0][0][0])
                shape = tuple(int(i) for i in entry[1][0])
                size = int(entry[2][0][0])
                server_time = float(entry[3][0][0])
                result_list.append((time_cost, shape, size, server_time))

            results_dict[key] = result_list

        results[var_name] = results_dict
    else:
        print(f"{var_name} not found in the .mat file.")




