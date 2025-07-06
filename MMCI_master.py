# File :     MMCI_master.py
# Author :   Chuxuan,Changjin
# Project :  SimDMMCI
import datetime
import os

import numpy as np
import scipy.io as sio
from tqdm import trange
import env.env_config as ec
import env.global_config as gc
import matplotlib.pyplot as plt
from Algorithm.Hiearachical_Q import Hierarchical_Q

nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
pwd = os.getcwd() + '/results/HQ/'+nowTime
os.makedirs(pwd)

Save_utility = np.zeros((gc.EPISODE_NUMBER, gc.STEP_NUMBER))
Save_latency = np.zeros((gc.EPISODE_NUMBER, gc.STEP_NUMBER))
Save_energy = np.zeros((gc.EPISODE_NUMBER, gc.STEP_NUMBER))
Save_accuracy = np.zeros((gc.EPISODE_NUMBER, gc.STEP_NUMBER))
Save_MasterAction = np.zeros((gc.EPISODE_NUMBER, gc.STEP_NUMBER))
Save_SubAction = np.zeros((gc.EPISODE_NUMBER, gc.STEP_NUMBER))


agent = Hierarchical_Q()
for episode in range(gc.EPISODE_NUMBER):
    master_policy = np.random.randint(0, len(ec.type_space))
    state = gc.get_state()
    for step in trange(gc.STEP_NUMBER):
        state_idx = agent.get_state_index(state)
        Save_MasterAction[episode, step] = master_policy
        a, nd = gc.get_action_type(master_policy)
        joint_state_tuple = (state_idx, master_policy)
        joint_state_masterPolicy = ec.State_masterPolicy_combination[joint_state_tuple]
        sub_policy = agent.getSubPolicy(joint_state_masterPolicy, step)
        Save_SubAction[episode, step] = sub_policy
        b = gc.get_action_layer(sub_policy)

        selected_time, selected_data, server_time = gc.get_selected_results(a, b, ec.results[state[1] - 1])
        latency_local = sum(selected_time)
        transmit_time = sum(selected_data) / ec.get_b(state[0])
        latency_server = sum(server_time)
        latency = latency_local + transmit_time + latency_server
        print(latency_local, transmit_time)
        energy_local = latency_local * ec.running_power
        energy_tranmission = transmit_time * ec.power * 10e-3
        energy = energy_local + energy_tranmission
        accuracy = gc.get_map(a)
        utility = - latency * gc.w1 - energy * gc.w2 + (accuracy - ec.mBase) * gc.w3


        next_h = gc.get_RSSI()  # reset
        next_task_num = gc.get_task_num()
        next_state = [next_h, next_task_num, latency, energy]
        next_state_idx = agent.get_state_index(next_state)

        agent.updateMasterPolicy(state_idx, master_policy, utility, next_state_idx)

        master_policy = agent.getMasterPolicy(next_state_idx, step)
        joint_state_tuple = (next_state_idx, master_policy)
        next_joint_state_masterPolicy = ec.State_masterPolicy_combination[joint_state_tuple]
        agent.updateSubPolicy(joint_state_masterPolicy, sub_policy, utility, next_joint_state_masterPolicy)
        state = next_state

        Save_utility[episode, step] = utility
        Save_latency[episode, step] = latency
        Save_energy[episode, step] = energy
        Save_accuracy[episode, step] = accuracy

    agent.reset()

x = np.arange(0, gc.STEP_NUMBER, 1)
utility = Save_utility.mean(axis=0)
delay = Save_latency.mean(axis=0)
energy = Save_energy.mean(axis=0)

sio.savemat(pwd + '/results.mat',
                {'utility': np.sum(Save_utility, 0) / gc.EPISODE_NUMBER,
                 'delay': np.sum(Save_latency, 0) / gc.EPISODE_NUMBER,
                 'energy': np.sum(Save_energy, 0) / gc.EPISODE_NUMBER,
                 'accuracy': np.sum(Save_accuracy, 0) / gc.EPISODE_NUMBER,
                 'Master':Save_MasterAction,
                 'Sub':Save_SubAction})

plt.figure()
plt.plot(x, utility)
plt.xlabel("time slot")
plt.ylabel("utility")
# plt.show()
# plt.savefig(pwd2 + "/utility.jpg")

plt.figure()
plt.plot(x, delay)
plt.xlabel("time slot")
plt.ylabel("delay")
# plt.show()
# plt.savefig(pwd2 + "/delay.jpg")

plt.figure()
plt.plot(x, energy)
plt.xlabel("time slot")
plt.ylabel("energy")
# plt.show()






