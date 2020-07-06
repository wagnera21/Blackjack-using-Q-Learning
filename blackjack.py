"""
Alexandra Wagner
CSCI 316
Spring 2020
Blackjack environment to test QL Agent
"""

import ql
import gym
import matplotlib.pyplot as plt
import numpy as np

NEPS = 10000
ALPHA = .1
GAMMA = .1
EPSILON = .1

env = gym.make('Blackjack-v0')

agent = ql.QLAgent(env, NEPS,ALPHA,GAMMA,EPSILON)
#agent.play()
agent.play_trained()

plt.clf()
plt.plot(agent.getAverageReward())
plt.xlabel('samples')
plt.ylabel('average reward')
plt.show()

env.close()
