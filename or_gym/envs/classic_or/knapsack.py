import numpy as np
import gym

class KnapsackEnv(gym.Env):

    def __init__(self):
        print('Knapsack Env initialized')
    def step(self):
        print('Step successful')
    def reset(self):
        print('Knapsack emptied')
