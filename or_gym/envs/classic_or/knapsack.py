import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
import copy

class KnapsackEnv(gym.Env):
    '''
    Unbounded Knapsack Problem

    The Knapsack Problem (KP) is a combinatorial optimization problem which
    requires the user to select from a range of goods of different values and
    weights in order to maximize the value of the selected items within a 
    given weight limit. This version is unbounded meaning the we can select
    items without limit. 

    The episodes proceed by selecting items and placing them into the
    knapsack one at a time until the weight limit is reached or exceeded, at
    which point the episode ends.

    Observation:
        Type: Tuple, Discrete
        0: list of item weights
        1: list of item values
        2: current weight in knapsack

    Actions:
        Type: Discrete
        0: Place item 0 into knapsack
        1: Place item 1 into knapsack
        2: ...

    Reward:
        Value of item successfully placed into knapsack or 0 if the item
        doesn't fit, at which point the episode ends.

    Starting State:
        Lists of available items and empty knapsack.

    Episode Termination:
        Full knapsack or selection that puts the knapsack over the limit.
    '''
    
    def __init__(self):
        self.item_weights = [1, 2, 3, 6, 10, 18]
        self.item_values = [0, 1, 3, 14, 20, 100]
        self.item_numbers = np.arange(len(self.item_weights))
        self.N = len(self.item_weights)
        self.max_weight = 15
        self.current_weight = 0
        
        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.N),
            spaces.Discrete(self.N),
            spaces.Discrete(1)))
        
        self.seed()
        self.reset()
        
    def step(self, item):
        # Check that item will fit
        if self.item_weights[item] + self.current_weight <= self.max_weight:
            self.current_weight += self.item_weights[item]
            reward = self.item_values[item]
            if self.current_weight == self.max_weight:
                done = True
            else:
                done = False
        else:
            # End trial if over weight
            reward = 0
            done = True
            
        self._update_state(item)
        return self.state, reward, done, {}
    
    def _get_obs(self):
        return self.state
    
    def _update_state(self, item=None):
        self.state = (self.item_weights,
                      self.item_values, 
                      self.current_weight)
    
    def reset(self):
        self.current_weight = 0
        self._update_state()
        return self.state
    
    def sample_action(self):
        return np.random.choice(self.item_numbers)
