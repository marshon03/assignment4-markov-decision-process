import gym
import os
import numpy as np

import matplotlib.pyplot as plt


def frozenlake_q_learning():
    # Environment initialization
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'q_learning')
    env = gym.wrappers.Monitor(gym.make('FrozenLake-v0'), folder, force=True)

    # Q and rewards
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iterations = []

    # Parameters
    alpha = 0.75
    discount = 0.90
    episodes = 1000

    # Episodes
    for episode in range(episodes):
        # Refresh state
        state = env.reset()
        done = False
        t_reward = 0
        max_steps = env.spec.max_episode_steps

        # Run episode
        for i in range(max_steps):
            if done:
                break

            current = state
            action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))

            state, reward, done, info = env.step(action)
            t_reward += reward
            Q[current, action] += alpha * (reward + discount * np.max(Q[state, :]) - Q[current, action])

        rewards.append(t_reward)
        iterations.append(i)

    # Close environment
    env.close()

    size = episodes / 50
    chunks = list(chunk_list(rewards, size))
    averages = [sum(chunk) / len(chunk) for chunk in chunks]

    plt.plot(range(0, len(rewards), int(size)), averages)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()


# Plot results
def chunk_list(l, n):
    for i in range(0, len(l), int(n)):
        yield l[i:i + int(n)]