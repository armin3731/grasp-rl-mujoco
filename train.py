# import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import QT, TrainModel

# * Settings ==============================================================
# if GPU is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
# EPISODES is the total number of episodes
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
TS = {
    "BATCH_SIZE": 128,
    "GAMMA": 0.99,
    "EPS_START": 0.9,
    "EPS_END": 0.05,
    "EPS_DECAY": 200,
    "TAU": 0.005,
    "LR": 1e-2,
}
# episodes to train
EPISODES = 1000
# The number of actions for robotic hand
N_ACTIONS = 2
# The the number of state observations (POSITION, ANGLE)
N_OBSERVATIONS = 2


# *Defining Finger Models =============================================
# Define the Networks for eahc finger
index_QT = QT(N_OBSERVATIONS, N_ACTIONS).to(device)

# Define the Optimazers for each finger
index_OPTIMIZER = optim.AdamW(index_QT.parameters(), lr=TS["LR"], amsgrad=True)

# Define Training Model for each finger
index_TR_Model = TrainModel(index_QT, device, N_ACTIONS, TS)


# * Training ==========================================================
# TODO: Training should move into python mujoco file
for episode in range(EPISODES):
    # state = environment.reset()

    # Choose the action with the highest value in the current state
    # TODO: State should be crated from mujoco
    object = {
        "position": 2.32,
        "angle": 14.3,
    }

    current_state = [object["position"], object["angle"]]
    print("current_state------------------", current_state)
    index_action = index_TR_Model.select_action(current_state, episode)

    # Implement this action and move the agent in the desired direction
    # TODO: Response of the action from mujoco
    reward = 1

    # Update Q(s,a)
    index_TR_Model.optimize(current_state, index_action, reward)

    print("---------------------------------------")
    print("Training Happend")
