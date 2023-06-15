import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('Taxi-v3') # I used version 0.21.0.

#0 -> Down
#1 -> Up
#2 -> Right
#3 -> left
#4 -> Pickup
#5 -> Drop
#6 -> Up-East
#7 -> Up- West
#8 -> Down- East
#9 -> Down- West
class ExtraActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(10)
        self.taxi_row, self.taxi_col, self.pass_idx, self.dest_idx = self.env.env.decode(self.env.env.s)
        self.state = self.env.env.encode(self.taxi_row, self.taxi_col, self.pass_idx, self.dest_idx)

    def step(self, action):
        # Map new actions to existing actions
        if action == 6:  # Up-East
            next_state, reward, done, info = self.env.step(1)
            if  self.state == next_state:
                return next_state, reward, done, info
            two_next_state, reward, done, info = self.env.step(2)
            if next_state == two_next_state:
                i,j,k,d=self.env.step(0)
                return i,j,k,d
            return two_next_state, reward, done, info
        elif action == 7:  # Up-West
            next_state, reward, done, info = self.env.step(1)
            if  self.state == next_state:
                return  next_state, reward, done, info
            two_next_state, reward, done, info = self.env.step(3)
            if next_state == two_next_state:
                i,j,k,d = self.env.step(0)
                return i,j,k,d
            return two_next_state, reward, done, info
        elif action == 8:  # Down-East
            next_state, reward, done, info = self.env.step(0)
            if  self.state == next_state:
                return next_state, reward, done, info
            two_next_state, reward, done, info = self.env.step(2)
            if next_state == two_next_state:
                i,j,k,d = self.env.step(1)
                return i,j,k,d
            return two_next_state, reward, done, info
        elif action == 9:  # Down-West
            next_state, reward, done, info = self.env.step(0)
            if  self.state == next_state:
                return next_state, reward, done, info
            two_next_state, reward, done, info = self.env.step(3)
            if next_state == two_next_state:
                i,j,k,d = self.env.step(1)
            return two_next_state, reward, done, info
        else:
            return self.env.step(action)
        
env = ExtraActionWrapper(env)

# initialize the Q-table to zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# set hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 10000
# Saving reward and episode for calculating cumulative reward
rewards = []
episode = []
cumulative_reward = 0
# define the epsilon-greedy policy
def epsilon_greedy(state):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    return action

# run the SARSA algorithm
for epi in range(num_episodes):
    state = env.reset()
    done = False
    cumulative_reward = 0
    
    while not done:
        # choose an action using the epsilon-greedy policy
        action = epsilon_greedy(state)
        
        # take the chosen action and observe the next state and reward
        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward
        next_action = epsilon_greedy(next_state)
        
        # update the Q-value for the current state and action
        q_table[state][action] += alpha * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])
        
        state = next_state

    rewards.append(cumulative_reward)
    episode.append(epi)

'''
Calculating the cumulative reward for each episode after the agent has been trained.
You can run the following code to see how well the agent performs after learning Q-values using SARSA.
'''

# state = env.reset()
# done = False

# for ep in range (100):
#     cumulative_reward = 0
#     state = env.reset()
#     done = False
#     while not done:
#         action = np.argmax(q_table[state])
#         next_state, reward, done, info = env.step(action)
#         state = next_state
#         cumulative_reward += reward
#     rewards.append(cumulative_reward)
#     episode.append(ep)

plt.plot(episode, rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('َُSARSA Performance')
plt.show()