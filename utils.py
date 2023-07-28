import random
import math
import numpy as np
from scipy.spatial.transform import Rotation
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import mujoco as mj


BEND_FORCE = 29.5  # The force that a finger can produce when bending


class QT(nn.Module):
    """
    A class to create QT network
    """

    def __init__(self, n_observations: int, n_actions: int, name: str):
        super(QT, self).__init__()
        print(
            "n_actions,n_observations<<<<<<<<<<<<<<<<<<<<<<<<<",
            n_actions,
            n_observations,
        )
        self.layer1 = nn.Linear(n_observations, 6, dtype=torch.float64)
        self.layer2 = nn.Linear(6, 15, dtype=torch.float64)
        self.layer3 = nn.Linear(15, n_actions, dtype=torch.float64)
        self.name = name

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """torch default function"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def save_parameters(self, path: str):
        torch.save(self.state_dict(), os.path.join(path, "%s.pt" % (self.name)))

    def load_parameters(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def predict(self, state):
        tensor_state = torch.tensor(state, requires_grad=False)
        _, action = self.forward(tensor_state).max(0)
        return action


class TrainModel:
    """A class that trains the model"""

    def __init__(
        self,
        policy_net: QT,
        device: torch.device,
        n_actions: int,
        settings: dict,
        criterion=torch.nn.SmoothL1Loss(),
    ) -> None:
        self.policy_net = policy_net
        self.device = device
        self.n_actions = n_actions
        self.settings = settings
        self.criterion = criterion
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.settings["LR"], amsgrad=True
        )

    def select_action(self, state, episode: int):
        """This method choses the action. At first all actions are random. but after some episodes, actions are chosen from Q-Table"""
        sample = random.random()
        eps_threshold = self.settings["EPS_END"] + (
            self.settings["EPS_START"] - self.settings["EPS_END"]
        ) * math.exp(-1.0 * episode / self.settings["EPS_DECAY"])

        if sample > eps_threshold:
            print("EPSILON won !!!!!!!!!!!!!!!!!!!!!!!!!!!")
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                tensor_state = torch.tensor(state, requires_grad=False)
                print(
                    "state, torch_state ===============================",
                    state,
                    tensor_state,
                )
                policy_response = self.policy_net(tensor_state)
                highVal, action = self.policy_net(tensor_state).max(0)
                print(
                    "highVal, action, policy_res ===============================",
                    highVal,
                    action,
                    policy_response,
                )
                return action
        else:
            print("Sample Random won !!!!!!!!!!!!!!!!!!!!!!!!!!!")
            random_value = math.floor(random.random() * 2)
            if random_value > 1 or random_value < 0:
                random_value = 1
            return torch.tensor(
                [[random_value]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize(self, state, action_taken, reward):
        """
        This function optimize the network for one step
        """

        print("I am in Optimize =-=-=-=-=-=-=-=--------------------")
        tensor_state = torch.tensor(state, requires_grad=False)
        print(
            "state,tensor_state, action_taken, reward =-=-=-=-=-=-=-=--------------------",
            state,
            tensor_state,
            action_taken,
            reward,
        )

        current_actions = self.policy_net(tensor_state)
        print("output of the newort-=-=-=-=--==-=-=-=-=-=--==-")
        print("action_taken", action_taken)
        print("current_actions", current_actions)

        expected_actions = current_actions.clone()
        expected_actions[action_taken] = reward
        print("output of the newort-=-=-=-=--==-=-=-=-=-=--==-")
        print("current_actions", current_actions)
        print("expected_actions", expected_actions)
        print("output of the newort-=-=-=-=--==-=-=-=-=-=--==-")
        loss = self.criterion(current_actions, expected_actions)
        print("loss Function-=-=-=-=--==-=-=-=-=-=--==-")
        print(loss)

        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # *  some part of the optimization algorithm is based on the repo below
    # *  LOOK AT THIS -> https://github.com/hamedmokazemi/DeepQLearning_FrozenLake_1/blob/main/main_4x4.py


def random_pose(mj_data):
    """
    This function makes the object rotate and move randomly at
    fist moment of each simulation

    This numbers are chosen due to simulation dimensions
    change them carefully!
    """
    mj_data.qpos[14] = (random.random() * 1.1) - 0.4  # X
    mj_data.qpos[16] = (random.random() * 1.0) + 0.8  # Z
    rot = Rotation.from_euler(
        "xyz", np.array([0, ((random.random() * 60) - 30), 0]), degrees=True
    )
    mj_data.qpos[17:21] = rot.as_quat()

    return mj_data


def finger_bend(
    finger_num: int, action: int, mj_model, mj_data, bend_force: float = BEND_FORCE
):
    if action == 1:
        mj_model.actuator_gainprm[finger_num, 0] = bend_force
        mj_data.ctrl[finger_num] = 1 * np.pi
    return mj_model, mj_data


def mujoco_reset_env(mj_model, mj_data):
    mj.mj_resetData(mj_model, mj_data)
    mj_data = random_pose(mj_data)
    mj.mj_forward(mj_model, mj_data)
    return mj_model, mj_data
