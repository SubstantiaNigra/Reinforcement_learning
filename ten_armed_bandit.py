import numpy as np
import matplotlib.pyplot as plt
from random import random

def get_agent_and_env(mean, std, size: tuple, steps_num, starting_reward=0):
    rows, cols = size
    Q_values = np.ones(size) * starting_reward
    rng = np.random.default_rng()

    q_values = rng.normal(loc=mean, scale=std, size=size)
    Rewards = rng.normal(loc=q_values, scale=std, size=size)

    actions = np.arange(size[0])
    num_of_action = np.zeros(size) # count each action chosen

    t_steps = [j for j in range(steps_num)]
    return Q_values, q_values, Rewards, actions, num_of_action, t_steps

def make_choices(epsilon, steps_num, runs):
    random_choice = [random() for i in range(runs)]
    random_choice = np.asarray(random_choice)

    exploration_steps = np.where(random_choice >= (1 - epsilon))
    exploatation_steps = np.where(random_choice < (1 - epsilon))
    return exploration_steps[0], exploatation_steps[0]

def be_greedy(Q_vals, Rewards, exploatation_steps, num_of_action):
    max_Q_value_idx = np.argmax(Q_vals[:, exploatation_steps], axis=0)
    max_Q_value = np.zeros((rows, runs))
    for idx, given_run in enumerate(exploatation_steps):
        action_idx = max_Q_value_idx[idx]
        # max_Q_value[action_idx, given_run] = Rewards[action_idx, given_run]
        num_of_action[action_idx, given_run] += 1
        update = Rewards[action_idx, given_run] / num_of_action[action_idx, given_run]
        Q_vals[action_idx, given_run] += update
    return Q_vals, num_of_action

def explore(Q_vals, Rewards, exploration_steps, exploration_len, rows, runs, num_of_action):
    rng = np.random.default_rng()
    # choose random actions for given runs
    rnd_size = exploration_len # equal to len(exploration_steps)
    rnd_Q_value_idx = np.random.randint(rows, size=rnd_size) # indices of given
                                                # action for given run
    rnd_Q_value = np.zeros((rows, runs))
    # print('rnd_size', rnd_size)
    # print(rnd_Q_value_idx)

    # choose given actions and receive rewards
    for idx, given_run in enumerate(exploration_steps):
        action_idx = rnd_Q_value_idx[idx]
        # rnd_Q_value[action_idx, given_run] = Rewards[action_idx, given_run]
        num_of_action[action_idx, given_run] += 1
        update = Rewards[action_idx, given_run] / num_of_action[action_idx, given_run]
        Q_vals[action_idx, given_run] += update
    # print(rnd_Q_value)
    return Q_vals, num_of_action

rows = 10 # number of actions
runs = 1000 # number of runs
mean = 0
variance = 1
std = np.sqrt(variance)
size = (rows, runs)

steps_num = 1000
epsilon = 0.01 # probability of choosing random action

# create env and agent
Q_values, q_values, Rewards, actions, num_of_action, t_steps = get_agent_and_env(
                                                        mean, std, size, steps_num)

means_to_plot = np.zeros(steps_num)
step = 1
while step <= steps_num:
    # make choices whether to explore or to exploatate in a given run
    exploration_steps, exploatation_steps = make_choices(epsilon, steps_num, runs)
    exploatation_len = len(exploatation_steps)
    exploration_len = len(exploration_steps)

    if exploration_len != 0:
        Q_values, num_of_action = explore(Q_values, Rewards,
                exploration_steps, exploration_len, rows, runs, num_of_action)
    if exploatation_len != 0:
        Q_values, num_of_action = be_greedy(Q_values,
                                    Rewards, exploatation_steps, num_of_action)


    maximal_Q_values_idx = np.argmax(Q_values, axis=0)
    averages = np.mean(Q_values[maximal_Q_values_idx, :])
    means_to_plot[step-1] = averages
    step += 1



epsilon2 = 0.1 # probability of choosing random action
# create env and agent
Q_values, q_values, Rewards, actions, num_of_action, t_steps = get_agent_and_env(
                                                        mean, std, size, steps_num)
means_to_plot2 = np.zeros(steps_num)
step = 1
while step <= steps_num:
    # make choices whether to explore or to exploatate in a given run
    exploration_steps, exploatation_steps = make_choices(epsilon2, steps_num, runs)
    exploatation_len = len(exploatation_steps)
    exploration_len = len(exploration_steps)

    if exploration_len != 0:
        Q_values, num_of_action = explore(Q_values, Rewards,
                exploration_steps, exploration_len, rows, runs, num_of_action)
    if exploatation_len != 0:
        Q_values, num_of_action = be_greedy(Q_values,
                                    Rewards, exploatation_steps, num_of_action)


    maximal_Q_values_idx = np.argmax(Q_values, axis=0)
    averages = np.mean(Q_values[maximal_Q_values_idx, :])
    means_to_plot2[step-1] = averages
    step += 1


# plot average maximal values of Q_value
plt.figure()
plt.plot(range(steps_num), means_to_plot, 'r', label=f'epsilon={epsilon}')
plt.plot(range(steps_num), means_to_plot2, 'b', label=f'epsilon={epsilon2}')
plt.legend()
plt.show()
