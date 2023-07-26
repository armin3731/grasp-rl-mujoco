import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import QT, TrainModel

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We re-initialize the Q-table
# qtable = np.zeros((environment.observation_space.n, environment.action_space.n))


# Hyperparameters
# EPISODES is the total number of episodes
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
EPISODES = 1000
TS = {
    "BATCH_SIZE": 128,
    "GAMMA": 0.99,
    "EPS_START": 0.9,
    "EPS_END": 0.05,
    "EPS_DECAY": 200,
    "TAU": 0.005,
    "LR": 1e-4,
}

# The number of actions for robotic hand
n_actions = 2
# The the number of state observations
n_observations = 3


# Define the Networks for eahc finger
index_NET = QT(n_observations, n_actions).to(device)

# Define the Optimazers for each finger
index_OPTIMIZER = optim.AdamW(index_NET.parameters(), lr=TS["LR"], amsgrad=True)


# Define Training Model for each finger
index_TRAIN = TrainModel(index_NET, device, n_actions, TS)


# Training
# TODO: Training should move into python mujoco file
for episode in range(EPISODES):
    # state = environment.reset()

    # Choose the action with the highest value in the current state
    # TODO: State should be crated from mujoco
    object = {
        "position": 2.32,
        "angle": 14.3,
    }
    current_state = (object["position"], object["angle"])
    index_action = index_TRAIN.select_action(current_state, episode)

    # Implement this action and move the agent in the desired direction
    # TODO: Response of the action from mujoco
    new_state, reward, done, info = environment.step(index_action)

    # Update Q(s,a)
    index_TRAIN.optimize(reward)


print()
print("===========================================")
print("Q-table after training:")
print(qtable)

# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor("#efeeea")
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()
