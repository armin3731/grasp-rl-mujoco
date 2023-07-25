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
    "LR": 1e-4,
}

# Get number of actions from gym action space
n_actions = 2
# Get the number of state observations
#! state, info = env.reset()
n_observations = 3

index_NET = QT(n_observations, n_actions).to(device)
#! target_net = DQN(n_observations, n_actions).to(device)
#! target_net.load_state_dict(policy_net.state_dict())

index_OPTIMIZER = optim.AdamW(index_NET.parameters(), lr=TS["LR"], amsgrad=True)
#! memory = ReplayMemory(10000)


#! steps_done = 0


# episode_durations = []


# Hyperparameters
episodes = 1000  # Total number of episodes
# alpha = 0.5  # Learning rate
# gamma = 0.9  # Discount factor

# List of outcomes to plot
outcomes = []


index_TRAIN = TrainModel(index_NET, device, n_actions, TS)

# print("Q-table before training:")

# Training
for episode in range(episodes):
    # state = environment.reset()
    # done = False

    # By default, we consider our outcome to be a failure
    # outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    # while not done:
    # Choose the action with the highest value in the current state
    current_state = (POSITION, ANGLE, LENGTH)
    index_action = index_trian.select_action(current_state, episode)

    # If there's no best action (only zeros), take a random one
    # else:
    #     action = environment.action_space.sample()

    # Implement this action and move the agent in the desired direction
    new_state, reward, done, info = environment.step(action)

    # Update Q(s,a)
    index_train.optimize(reward)
    # qtable[state, action] = qtable[state, action] + alpha * (
    #     reward + gamma * np.max(qtable[new_state]) - qtable[state, action]
    # )

    # Update our current state
    # state = new_state

    # If we have a reward, it means that our outcome is a success
    # if reward:
    #     outcomes[-1] = "Success"

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


# def plot_durations(show_result=False):
#     plt.figure(1)
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())


def optimize_model():
    # if len(memory) < BATCH_SIZE:
    #     return
    # transitions = memory.sample(BATCH_SIZE)
    # # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # # detailed explanation). This converts batch-array of Transitions
    # # to Transition of batch-arrays.
    # batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print("Complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()


# Hyperparameters
episodes = 1000  # Total number of episodes
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor

# List of outcomes to plot
outcomes = []

print("Q-table before training:")
print(qtable)

# Training
for _ in range(episodes):
    state = environment.reset()
    done = False

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    # while not done:

    # Choose the action with the highest value in the current state
    # * Get from Policy Net
    if np.max(qtable[state]) > 0:
        action = np.argmax(qtable[state])

    # If there's no best action (only zeros), take a random one
    else:
        action = environment.action_space.sample()

    # Implement this action and move the agent in the desired direction
    new_state, reward, done, info = environment.step(action)

    # Update Q(s,a)
    qtable[state, action] = qtable[state, action] + alpha * (
        reward + gamma * np.max(qtable[new_state]) - qtable[state, action]
    )

    # Update our current state
    state = new_state

    # If we have a reward, it means that our outcome is a success
    if reward:
        outcomes[-1] = "Success"

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
