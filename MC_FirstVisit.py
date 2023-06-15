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
# Initialize the Q table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Initialize the returns and counts for each state-action pair
returns = {}
counts = {}
# Saving reward and episode for calculating cumulative reward
Rewards = []
Episode = []
cumulative_reward = 0
# Initialize the policy
def policy(observation, epsilon):
    if np.random.uniform() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[observation])

# Set hyperparameters
num_episodes = 10000
epsilon = 0.1
discount_factor = 0.9

# Run the Monte Carlo First-Visit algorithm
for epi in range(num_episodes):
    # Generate an episode
    episode = []
    observation = env.reset()
    done = False
    cumulative_reward = 0
    while not done:
        action = policy(observation, epsilon)
        next_observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        episode.append((observation, action, reward))
        observation = next_observation
        
    Rewards.append(cumulative_reward)
    Episode.append(epi)
    
    # Update Q table with the returns from the episode
    visited = set()
    for t, (s, a, r) in enumerate(episode):
        if (s, a) not in visited:
            visited.add((s,a))
            G = sum([r2 * (discount_factor ** d) for d, (s2, a2, r2) in enumerate(episode[t:])])
            if (s, a) in returns:
                returns[(s, a)].append(G)
            else:
                returns[(s, a)] = [G]
            counts[(s, a)] = counts.get((s, a), 0) + 1
            Q[s][a] = np.mean(returns[(s, a)])


'''
Calculating the cumulative reward for each episode after the agent has been trained.
You can run the following code to see how well the agent performs after learning Q-values using MC.
'''

# state = env.reset()
# done = False

# for ep in range (100):
#     cumulative_reward = 0
#     state = env.reset()
#     done = False
#     while not done:
#         action = np.argmax(Q[state])
#         next_state, reward, done, info = env.step(action)
#         state = next_state
#         cumulative_reward += reward
#     Rewards.append(cumulative_reward)
#     Episode.append(ep)

plt.plot(Episode, Rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('MC First Visit Control Performance')
plt.show()