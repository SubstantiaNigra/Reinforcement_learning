import numpy as np
import matplotlib.pyplot as plt
from random import random

def get_agent_and_env(mean, std, size: tuple, steps_num, starting_reward=0):
    rows, cols = size
    Q_values = np.ones(size) * starting_reward
    rng = np.random.default_rng()

    q_values = rng.normal(loc=mean, scale=std, size=size)

    actions = np.arange(size[0])
    num_of_action = np.zeros(size) # count each action chosen

    t_steps = [j+1 for j in range(steps_num)]
    return Q_values, q_values, actions, num_of_action, t_steps

def choose_action(size, epsilon, Q_values, actions):
    rng = np.random.default_rng()
    exp_probs = np.random.uniform(low=0, high=1, size=size[1])# probs to determine if
                                            #exploration or exploitation occurs
    explore = tuple(np.where(exp_probs > (1 - epsilon))) # indices of runs with exploration
                                                    # includes a case epsilon = 0
    exploit = tuple(np.where(exp_probs <= (1 - epsilon))) # indices of runs with exploitation
    chosen_actions = np.zeros(size[1]) # prepare a vector with chosen actions
    if explore[0].size: # if explore is not empty
        random_actions = rng.choice(actions, size=len(explore[0]))
        chosen_actions[explore] = random_actions
        chosen_actions = chosen_actions.astype(int)
    if exploit[0].size: # if exploit is not empty
        chosen_actions[exploit] = np.argmax(Q_values[:, exploit], axis=0)
        chosen_actions = chosen_actions.astype(int)
    return chosen_actions

def get_rewards(Q_values, actions, chosen_actions, n_A, num_of_action):
    rng = np.random.default_rng()
    Rewards = rng.normal(loc=q_values, scale=std, size=size) # choose rewards from
                                                            # q_values distributions
    current_reward = np.zeros(size[1]) # list to assign the most recent rewards
                                            # for averaging
    for run, action in enumerate(chosen_actions):
        num_of_action[action, run] += 1 # update the number of occurences of a choice
        Q_value = Q_values[action, run] # a Q_value we want to update
        Reward = Rewards[action, run] # Reward corresponding to Q_value
        diff = Reward - Q_value
        divider = 1/(num_of_action[action, run]) # how many times an action was chosen
        reward = Q_value + divider*diff # a value with which we want to update
                                        # Q_value for a given action
        Q_values[action, run] = reward
        current_reward[run] = reward # assign to a list for plotting
    return Q_values, num_of_action, current_reward

rows = 10 # number of actions
runs = 2000 # number of runs
size = (rows, runs)
mean = 0
variance = 1
std = np.sqrt(variance)


steps_num = 1000
epsilon = 0.01 # probability of choosing random action
epsilon2 = 0.1
epsilon3 = 0.0

# create env and agent
Q_values, q_values, actions, num_of_action, t_steps = get_agent_and_env(
                                                        mean, std, size, steps_num)

means_to_plot = np.zeros(steps_num)
means_to_plot[0] = 0 # initial average reward value
n_A = 1
while n_A < steps_num:
    chosen_actions = choose_action(size, epsilon, Q_values,
                                                actions)
    Q_values, num_of_action, current_reward = get_rewards(Q_values, actions, chosen_actions, n_A,
                            num_of_action)

    # obtain averages to create plot
    averages = np.mean(current_reward)
    means_to_plot[n_A] = averages

    n_A += 1

# plot average maximal values of Q_value
plt.figure()
plt.plot(range(steps_num), means_to_plot, 'r', label=f'epsilon={epsilon}')



# create env and agent
Q_values, q_values, actions, num_of_action, t_steps = get_agent_and_env(
                                                        mean, std, size, steps_num)

means_to_plot2 = np.zeros(steps_num)
means_to_plot2[0] = 0 # initial average reward value
n_A = 1
while n_A < steps_num:
    chosen_actions = choose_action(size, epsilon2, Q_values, actions)
    Q_values, num_of_action, current_reward = get_rewards(Q_values, actions, chosen_actions, n_A,
                            num_of_action)

    averages = np.mean(current_reward)
    means_to_plot2[n_A] = averages

    n_A += 1

# plot average maximal values of Q_value
plt.plot(range(steps_num), means_to_plot2, 'b', label=f'epsilon={epsilon2}')



# create env and agent
Q_values, q_values, actions, num_of_action, t_steps = get_agent_and_env(
                                                        mean, std, size, steps_num)
means_to_plot3 = np.zeros(steps_num)
means_to_plot3[0] = 0 # initial average reward value
n_A = 1
while n_A < steps_num:
    chosen_actions = choose_action(size, epsilon3, Q_values, actions)
    Q_values, num_of_action, current_reward = get_rewards(Q_values, actions, chosen_actions, n_A,
                            num_of_action)

    averages = np.mean(current_reward)
    means_to_plot3[n_A] = averages

    n_A += 1

# plot average maximal values of Q_value
plt.plot(range(steps_num), means_to_plot3, 'g', label=f'epsilon={epsilon3}')


plt.legend()
plt.show()
