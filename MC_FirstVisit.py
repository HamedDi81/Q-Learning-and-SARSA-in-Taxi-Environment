import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3') # I used version 0.21.0.


# Initialize the Q table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Initialize the returns and counts for each state-action pair
returns = {}
counts = {}

# Initialize the policy
def policy(observation, epsilon):
    if np.random.uniform() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[observation])

# Set hyperparameters
num_episodes = 100000
epsilon = 0.1
discount_factor = 0.9

# Run the Monte Carlo First-Visit algorithm
for i in range(num_episodes):
    # Generate an episode
    episode = []
    observation = env.reset()
    done = False
    while not done:
        action = policy(observation, epsilon)
        next_observation, reward, done, info = env.step(action)
        episode.append((observation, action, reward))
        observation = next_observation
    
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

#Calculating the cumulative reward for each episode
Rewards = []
Episode = []
cumulative_reward = 0
state = env.reset()
done = False

for ep in range (100):
    cumulative_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        state = next_state
        cumulative_reward += reward
    Rewards.append(cumulative_reward)
    Episode.append(ep)

plt.plot(Episode, Rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('MC First Visit Control Performance')
plt.show()