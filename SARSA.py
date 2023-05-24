import gym
import numpy as np

env = gym.make('Taxi-v3') # I used version 0.21.0.

# initialize the Q-table to zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# set hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 100000

# define the epsilon-greedy policy
def epsilon_greedy(state):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    return action

# run the SARSA algorithm
for i in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # choose an action using the epsilon-greedy policy
        action = epsilon_greedy(state)
        
        # take the chosen action and observe the next state and reward
        next_state, reward, done, info = env.step(action)
        next_action = epsilon_greedy(next_state)
        
        # update the Q-value for the current state and action
        q_table[state][action] += alpha * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])
        
        state = next_state