"""
Alexandra Wagner
CSCI 316
Spring 2020
QLAgent Class
"""

import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import json

class QLAgent():
    def __init__(self, env, alpha=0.75, gamma=0.8, epsilon=0.9, episodes=100000):
        self.env = env

        self.actions = list(range(self.env.action_space.n))

        self.Qtable = dict() 
        self.epsilon = epsilon   
        self.alpha = alpha      
        self.gamma = gamma       

        self.episodes = episodes 
        self.small_decrement = (0.1 * epsilon) / (0.3 * episodes) 
        self.big_decrement = (0.8 * epsilon) / (0.4 * episodes)

        self.epsiodes_left = episodes

        self.rounds = 800
        self.samples = 50

        self.average_reward = []

    def updateParam(self):
        if self.epsiodes_left > 0.7 * self.episodes:
            self.epsilon -= self.small_decrement
        elif self.epsiodes_left > 0.3 * self.episodes:
            self.epsilon -= self.big_decrement
        elif self.epsiodes_left > 0:
            self.epsilon -= self.small_decrement
        else:
            self.epsilon = 0.0
            self.alpha = 0.0

        self.epsiodes_left -= 1

    def newQ(self, state):
        if state not in self.Qtable:
            self.Qtable[state] = dict((action, 0.0) for action in self.actions)

    def get_maxQ(self, state):
        self.newQ(state)
        return max(self.Qtable[state].values())

    def choose_action(self, state):
        self.newQ(state)
        if random.random() > self.epsilon:
            maxQ = self.get_maxQ(state)
            action = random.choice([k for k in self.Qtable[state].keys()
                                    if self.Qtable[state][k] == maxQ])
        else:
            action = random.choice(self.actions)
        self.updateParam()
        return action


    def play_trained(self):
        self.average_reward= []
        state = self.env.reset()

        for sample in range(self.samples):
            round = 1
            total_reward = 0 
            
            while round <= self.rounds:
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.Qtable[state][action] += self.alpha * (reward + (self.gamma * self.get_maxQ(next_state)) - self.Qtable[state][action])
                total_reward += reward
                if done:
                    self.env.reset()
                    round += 1
                    next_state, reward, done, info = self.env.step(action)
                    self.average_reward.append(total_reward/(sample*self.rounds + round))

        print("Training finished.\n")
    
    
    def play(self):

        self.average_reward= []
        state = self.env.reset()

        for sample in range(self.samples):
            round = 1
            total_reward = 0 
            
            while round <= self.rounds:
                action = self.env.action_space.sample()  
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    self.env.reset()
                    round += 1
            self.average_reward.append(total_reward)
            #print('Episode ', sample, 'Average Reward =', self.average_reward)

            
    def getNeps(self):
        return self.samples

    def getQtable(self):
        return self.Qtable

    def getAverageReward(self):
        return self.average_reward
        
        

