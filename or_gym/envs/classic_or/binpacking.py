import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
import copy


BIG_NEG_REWARD = -100
# Matches the AWS model as first pass
class BinPackingEnv(gym.Env):
    '''
    Online Bin Packing Problem

    The Bin Packing Problem (BPP) is a combinatorial optimization problem which
    requires the user to select from a range of goods of different values and
    weights in order to maximize the value of the selected items within a 
    given weight limit. This version is online meaning each item is randonly
    presented to the algorithm one at a time, at which point the algorithm 
    can either accept or reject the item. After seeing a fixed number of 
    items are shown, the episode terminates. If the weight limit is reached
    before the episode ends, then it terminates early.

    Observation:
        Type: Tuple, Discrete
        0 - bin_capacity: Count of bins at a given level h
        -1: Current item size


    Actions:
        Type: Discrete
        0: Open a new bin and place item into bin
        1+: Attempt to place item into bin at the given level

    Reward:
        Negative of the waste, which is the difference between the current
        size and excess space of the bin.

    Starting State:
        No available bins and random starting item
        
    Episode Termination:
        When invalid action is selected (e.g. attempt to place item in non-existent
        bin), bin limits are exceeded, or step limit is reached.
    '''
    def __init__(self):
        self.bin_capacity = 9
        self.item_sizes = [2, 3]
        self.item_probs = [0.8, 0.2]
        self.step_counter = 0
        self.step_limit = 1000
        
        self.observation_space = spaces.Box(
            low=np.array([0] * (1 + self.bin_capacity)),
            high=np.array([self.step_limit] * self.bin_capacity + [max(self.item_sizes)]),
            dtype=np.uint32)
        
        self.action_space = spaces.Discrete(self.bin_capacity)
        
        self.seed()
        self.state = self.reset()
        
    def step(self, action):
        done = False
        if action >= self.bin_capacity:
            raise ValueError('{} is an invalid action. Must be between {} and {}'.format(
                action, 0, self.bin_capacity))
        elif action > (self.bin_capacity - self.item_size):
            # Bin overflows
            reward = BIG_NEG_REWARD - self.waste
            done = True
        elif action == 0:
            # Create new bin
            self.bin_levels[self.item_size] += 1
            # This waste penalty seems very strange, it only occurs
            # when a new bin is opened.
            self.waste = self.bin_capacity - self.item_size
            reward = -1 * self.waste
        elif self.bin_levels[action] == 0:
            # Can't insert item into non-existent bin
            reward = BIG_NEG_REWARD - self.waste
            done = True
        else:
            if action + self.item_size == self.bin_capacity:
                self.num_full_bins += 1
            else:
                self.bin_levels[action + self.item_size] += 1
            self.waste = -self.item_size
            reward = -1 * self.waste
            
            self.bin_levels[action] -= 1
        
        self.total_reward += reward
        
        if self.step_counter >= self.step_limit:
            done = True
            
        if self.step_counter == self.step_limit:
            done = True
            
        self.item_size = self.get_item()
        state = self.bin_levels + [self.item_size]
        
        return self.state, reward, done, {}
    
    def get_item(self):
        return np.random.choice(self.item_sizes, p=self.item_probs)
        
    def sample_action(self):
        return self.action_space.sample()
    
    def reset(self):
        self.current_weight = 0
        self.step_counter = 0        
        self.num_full_bins = 0
        self.total_reward = 0
        self.waste = 0
        self.step_counter = 0
        
        self.bin_levels = [0] * self.bin_capacity
        self.item_size = self.get_item()
        initial_state = self.bin_levels + [self.item_size]
        return initial_state