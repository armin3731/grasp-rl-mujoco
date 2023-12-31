import torch

from utils import QT, TrainModel, random_pose, finger_bend, mujoco_reset_env

import mujoco as mj
import numpy as np
import os
import math


# * Settings ==============================================================
# folder to save the model at the end
SAVE_FOLDER = "models/"

# Mujoco Settings
MJ_XML_PATH = "RoboticHand.xml"
TIME_TO_HOLD = 1.5  # The time that robotic hand should keep the object (is Sec)

EPISODES = 5  # EPISODES is the total number of episodes
N_ACTIONS = 2  # The number of actions for robotic hand
N_OBSERVATIONS = 6  # The the number of state observations [TIP_POSITION[x y z], END_POSITION[[x y z]]]

# Hyperparameters
HP = {
    "EPS_START": 0.9,  # EPS_START is the starting value of epsilon
    "EPS_END": 0.05,  # EPS_END is the final value of epsilon
    "EPS_DECAY": math.floor(
        EPISODES / 5
    ),  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    "LR": 1e-2,  # LR is the learning rate of the ``AdamW`` optimizer
}


# * Defining Finger QT Models =============================================
# if GPU is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the Networks for each finger
index_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="index_finger").to(device)
middle_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="middle_finger").to(device)
ring_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="ring_finger").to(device)
pinky_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="pinky_finger").to(device)
thumb_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="thumb_finger").to(device)

# Define Training Model for each finger
index_TR_Model = TrainModel(index_QT, device, N_ACTIONS, HP)
middle_TR_Model = TrainModel(middle_QT, device, N_ACTIONS, HP)
ring_TR_Model = TrainModel(ring_QT, device, N_ACTIONS, HP)
pinky_TR_Model = TrainModel(pinky_QT, device, N_ACTIONS, HP)
thumb_TR_Model = TrainModel(thumb_QT, device, N_ACTIONS, HP)


# * Mujoco Functions =======================================================
def controller(
    mj_model,
    mj_data,
    actions,
):
    # put the controller here
    for each_finger_idx in range(5):
        mj_model, mj_data = finger_bend(
            each_finger_idx, actions[each_finger_idx], mj_model, mj_data
        )


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
    print("Episode %i of %i" % (episode + 1, EPISODES))
    mj_model, mj_data = mujoco_reset_env(mj_model, mj_data)  # reset mujoco env
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
    print("current_state : ", current_state)
    # Select actions
    index_action = index_TR_Model.select_action(current_state, episode)
    middle_action = middle_TR_Model.select_action(current_state, episode)
    ring_action = ring_TR_Model.select_action(current_state, episode)
    pinky_action = pinky_TR_Model.select_action(current_state, episode)
    thumb_action = thumb_TR_Model.select_action(current_state, episode)

    actions = [index_action, middle_action, ring_action, pinky_action, thumb_action]
    print("actions : ", actions)
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

    # Reward calculation
    # TODO: Reward should be finger dependent
    reward = is_holding
    print("reward : ", reward)

    # Update Q-Tables
    index_TR_Model.optimize(current_state, index_action, reward)
    middle_TR_Model.optimize(current_state, middle_action, reward)
    ring_TR_Model.optimize(current_state, ring_action, reward)
    pinky_TR_Model.optimize(current_state, pinky_action, reward)
    thumb_TR_Model.optimize(current_state, index_action, reward)

    print("--------------------------------------------")

print("Training Done")
index_QT.save_parameters(SAVE_FOLDER)
middle_QT.save_parameters(SAVE_FOLDER)
ring_QT.save_parameters(SAVE_FOLDER)
pinky_QT.save_parameters(SAVE_FOLDER)
thumb_QT.save_parameters(SAVE_FOLDER)
print("QT-Models Saved Successfully")
