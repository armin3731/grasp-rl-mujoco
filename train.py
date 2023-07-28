import torch
import torch.nn as nn
import torch.optim as optim

from utils import QT, TrainModel

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math

# * Settings ==============================================================
# Mujoco Settings
MJ_XML_PATH = "RoboticHand.xml"
TIME_TO_HOLD = 1.5  # The time that robotic hand should keep the object (is Sec)
BEND_FORCE = 29.5  # The force that a finger can produce when bending

# if GPU is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPISODES = 10  # EPISODES is the total number of episodes
N_ACTIONS = 2  # The number of actions for robotic hand
N_OBSERVATIONS = 6  # The the number of state observations [TIP_POSITION[x y z], END_POSITION[[x y z]]]

# Hyperparameters
HP = {
    # "BATCH_SIZE": 128,# BATCH_SIZE is the number of transitions sampled from the replay buffer
    # "GAMMA": 0.99,# GAMMA is the discount factor as mentioned in the previous section
    "EPS_START": 0.9,  # EPS_START is the starting value of epsilon
    "EPS_END": 0.05,  # EPS_END is the final value of epsilon
    "EPS_DECAY": math.floor(
        EPISODES / 5
    ),  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # "TAU": 0.005,# TAU is the update rate of the target network
    "LR": 1e-2,  # LR is the learning rate of the ``AdamW`` optimizer
}


# * Defining Finger QT Models =============================================
# Define the Networks for each finger
index_QT = QT(N_OBSERVATIONS, N_ACTIONS).to(device)

# Define Training Model for each finger
index_TR_Model = TrainModel(index_QT, device, N_ACTIONS, HP)


# * Mujoco Functions ===================================================
def controller(
    mj_model,
    mj_data,
    actions,
):
    # put the controller here
    index_idx = 0
    mj_model, mj_data = finger_bend(index_idx, actions[index_idx], mj_model, mj_data)


def finger_bend(
    finger_num: int, action: int, mj_model, mj_data, bend_force: float = BEND_FORCE
):
    if action == 1:
        mj_model.actuator_gainprm[finger_num, 0] = bend_force
        # model.actuator_biasprm[actuatorNum , 1] = -kp
        mj_data.ctrl[finger_num] = 1 * np.pi
    return mj_model, mj_data


def mujoco_reset_env(mj_model, mj_data):
    mj.mj_resetData(mj_model, mj_data)
    mj.mj_forward(mj_model, mj_data)


# * Initializations ======================================================
# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, MJ_XML_PATH)
xml_path = abspath
# MuJoCo data structures
mj_model = mj.MjModel.from_xml_path(MJ_XML_PATH)  # MuJoCo model
mj_data = mj.MjData(mj_model)  # MuJoCo data

# * Training ============================================================
for episode in range(EPISODES):
    mujoco_reset_env(mj_model, mj_data)  # reset mujoco env
    sim_start = mj_data.time
    tip_location = mj_data.site_xpos[0]  # Object's tip location
    end_location = mj_data.site_xpos[1]  # Object's end location
    object_properties = {
        "tip": np.squeeze(np.array(tip_location)),
        "end": np.squeeze(np.array(end_location)),
    }

    # Action from policy_net
    # Choose the action with the highest value in the current state
    current_state = np.double(
        np.append(object_properties["tip"], object_properties["end"])
    )
    print("current_state------------------", current_state)
    # Select actions
    index_action = index_TR_Model.select_action(current_state, episode)
    actions = [index_action]
    print("actions ---------------------", actions)
    # Inject actions into simulation
    mj.set_mjcb_control(controller(mj_model, mj_data, actions))

    # an indicator to show if the robotic hand is holding the object or not
    is_holding = 0

    # Start Simulation
    while True:
        mj.mj_step(mj_model, mj_data)

        # Stop conditions
        if mj_data.time - sim_start >= TIME_TO_HOLD:  # Success
            is_holding = 1
            break
        if tip_location[2] <= 0 or end_location[2] <= 0:  # Failure
            break

    print("Till HEREREREREREREREREREREREERERREREREERERERRR")
    # Reward calculation
    # TODO: Reward should be finger dependent
    reward = is_holding

    # Update Q-Tables
    index_TR_Model.optimize(current_state, index_action, reward)

    print("---------------------------------------")
    print("Episode Done")

print("Training Done")
# TODO: Save the model
