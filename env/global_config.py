# File :     global_config.py
# Author :   Chuxuan,Changjin
# Project :  SimDMMCI
import numpy as np
import scipy.io
from env.env_config import TaskNum, RssiSet, Type,M


EPISODE_NUMBER = 5
STEP_NUMBER = 3000

w1 = 1
w2 = 1
w3 = 0.1


def get_action_type(type_index, modal_num=6, type_len=2, node_len=1):
    types = []
    nodes = []
    base = type_len * node_len
    for _ in range(modal_num):
        index = type_index % base
        types.append(index // node_len)
        nodes.append(index % node_len)
        type_index //= base
    types.reverse()
    nodes.reverse()
    return types, nodes




def get_action_layer(layer_index, modal_num=6, layer_len=3):
    result = []
    for _ in range(modal_num):
        result.append(layer_index % layer_len)
        layer_index //= layer_len
    result.reverse()
    return result

def get_task_num():
    return np.random.choice(TaskNum)

def get_RSSI():
    return np.random.choice(RssiSet)

def get_state():
    h = get_RSSI()
    task_num = get_task_num()
    latency = 0.83
    energy = 2.0
    return [h, task_num, latency, energy]



def get_selected_results(a, b, results_dict):
    keys = ['vision', 'text', 'audio', 'depth', 'thermal', 'imu']
    selected_time = []
    selected_data = []
    selected_server_time = []

    for i in range(6):
        suffix = '100' if a[i] == 1 else '50'
        key = f"{keys[i]}_{suffix}"
        if key in results_dict:
            selected_time.append(results_dict[key][b[i]][0])
            selected_data.append(results_dict[key][b[i]][2])
            selected_server_time.append(results_dict[key][b[i]][3])


    return selected_time, selected_data, selected_server_time

acc_data = scipy.io.loadmat('map.mat')

def get_map(a):
    index = 0
    for b in a:
        index = index * M + b
    return acc_data[index]



