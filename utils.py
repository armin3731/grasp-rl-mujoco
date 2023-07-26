import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QT(nn.Module):
    """
    A class to create QT network
    """

    def __init__(self, n_observations, n_actions):
        super(QT, self).__init__()
        self.layer1 = nn.Linear(n_observations, 5)
        self.layer2 = nn.Linear(5, 15)
        self.layer3 = nn.Linear(15, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """torch default function"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


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
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[math.ceil(random.random() * 2)]], device=self.device, dtype=torch.long
            )

    def optimize(self, state, action_taken, reward):
        """
        This function optimize the network for one step
        """
        current_actions = self.policy_net(state)

        expected_actions = current_actions.copy()
        expected_actions[0][action_taken] = reward

        loss = self.criterion(current_actions, expected_actions)

        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # *  some part of the optimization algorithm is based on the repo below
    # *  LOOK AT THIS -> https://github.com/hamedmokazemi/DeepQLearning_FrozenLake_1/blob/main/main_4x4.py
