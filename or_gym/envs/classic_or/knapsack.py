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
    given weight limit. This version is unbounded meaning that we can select
    items without limit. 

    The episodes proceed by selecting items and placing them into the
    knapsack one at a time until the weight limit is reached or exceeded, at
    which point the episode ends.

    Observation:
        Type: Tuple, Discrete
        0: list of item weights
        1: list of item values
        2: maximum weight of the knapsack
        3: current weight in knapsack

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
        self.item_weights = weights
        self.item_values = values
        self.item_numbers = np.arange(len(self.item_weights))
        self.N = len(self.item_weights)
        self.max_weight = 200
        self.current_weight = 0
        
        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.N),
            spaces.Discrete(self.N),
            spaces.Discrete(2)))
        
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
            
        self._update_state()
        return self.state, reward, done, {}
    
    def _get_obs(self):
        return self.state
    
    def _update_state(self):
        self.state = (
            self.item_weights,
            self.item_values,
            self.max_weight,
            self.current_weight)
    
    def reset(self):
        self.current_weight = 0
        self._update_state()
        return self.state
    
    def sample_action(self):
        return np.random.choice(self.item_numbers)

class BoundedKnapsackEnv(KnapsackEnv):
    '''
    Bounded Knapsack Problem

    The Knapsack Problem (KP) is a combinatorial optimization problem which
    requires the user to select from a range of goods of different values and
    weights in order to maximize the value of the selected items within a 
    given weight limit. This version is bounded meaning each item can be
    selected a limited number of times.

    The episodes proceed by selecting items and placing them into the
    knapsack one at a time until the weight limit is reached or exceeded, at
    which point the episode ends.

    Observation:
        Type: Tuple, Discrete
        0: list of item weights
        1: list of item values
        2: list of item limits
        3: maximum weight of the knapsack
        4: current weight in knapsack


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
        self.item_limits_init = limits
        self.item_limits = self.item_limits_init.copy()
        super().__init__()
        
    def step(self, item):
        # Check item limit
        if self.item_limits[item] > 0:
            # Check that item will fit
            if self.item_weights[item] + self.current_weight <= self.max_weight:
                self.current_weight += self.item_weights[item]
                reward = self.item_values[item]
                if self.current_weight == self.max_weight:
                    done = True
                else:
                    done = False
            else:
                # End if over weight
                reward = 0
                done = True
        else:
            # End if item is unavailable
            reward = 0
            done = True
            
        self._update_state()
        return self.state, reward, done, {}
        
    def _update_state(self, item=None):
        if item is not None:
            self.item_limits[item] -= 1
            
        self.state = (
            self.item_weights,
            self.item_values,
            self.item_limits,
            self.max_weight,
            self.current_weight)
        
    def sample_action(self):
        return np.random.choice(
            self.item_numbers[np.where(self.item_limits!=0)])
    
    def reset(self):
        self.current_weight = 0
        self.item_limits = self.item_limits_init.copy()
        self._update_state()
        return self.state

class OnlineKnapsackEnv(BoundedKnapsackEnv):
    '''
    Online Knapsack Problem

    The Knapsack Problem (KP) is a combinatorial optimization problem which
    requires the user to select from a range of goods of different values and
    weights in order to maximize the value of the selected items within a 
    given weight limit. This version is online meaning each item is randonly
    presented to the algorithm one at a time, at which point the algorithm 
    can either accept or reject the item. After seeing a fixed number of 
    items are shown, the episode terminates. If the weight limit is reached
    before the episode ends, then it terminates early.

    Observation:
        Type: Tuple, Discrete
        0: list of item weights
        1: list of item values
        2: list of item limits
        3: maximum weight of the knapsack
        4: current weight in knapsack


    Actions:
        Type: Discrete
        0: Reject item
        1: Place item into knapsack

    Reward:
        Value of item successfully placed into knapsack or 0 if the item
        doesn't fit, at which point the episode ends.

    Starting State:
        Lists of available items and empty knapsack.

    Episode Termination:
        Full knapsack, selection that puts the knapsack over the limit, or
        the number of items to be drawn has been reached.
    '''
    def __init__(self):
        BoundedKnapsackEnv.__init__(self)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.N),
            spaces.Discrete(self.N),
            spaces.Discrete(2)))

        self.step_counter = 0
        self.step_limit = 50
        
        self.seed()
        self.state = self.reset()
        
    def step(self, action):
        # Check that item will fit
        if bool(action):
            # Check that item will fit
            if self.item_weights[self.current_item] + self.current_weight <= self.max_weight:
                self.current_weight += self.item_weights[self.current_item]
                reward = self.item_values[self.current_item]
                if self.current_weight == self.max_weight:
                    done = True
                else:
                    done = False
            else:
                # End if over weight
                reward = 0
                done = True
        else:
            reward = 0
            done = False
            
        self.step_counter += 1
        if self.step_counter >= self.step_limit:
            done = True
            
        self._update_state()
        return self.state, reward, done, {}
    
    def _update_state(self):
        self.current_item = np.random.choice(self.item_numbers, p=self.item_probs)
        self.state = (
            self.item_weights,
            self.item_values,
            self.item_probs,
            self.max_weight,
            np.array([
                self.current_weight,
                self.current_item
            ]))
        
    def sample_action(self):
        return np.random.choice([0, 1])
    
    def reset(self):
        if not hasattr(self, 'item_probs'):
            self.item_probs = self.item_limits_init / self.item_limits_init.sum()
        self.current_weight = 0
        self.step_counter = 0
        self.item_limits = self.item_limits_init.copy()
        self._update_state()
        return self.state

# Generated randomly with np.random.randint(30, 200)
values = np.array([28, 23, 16,  9, 20,  7, 22, 21, 23,  7, 21, 18, 23,  1, 17, 27,  0,
    1, 19, 10, 26, 28,  4, 11, 17,  9, 24,  7,  0, 16, 26, 16,  0,  3,
    0, 25, 18, 25,  7, 19, 17,  9, 12,  9, 24,  4, 12,  4, 23,  4,  8,
    15, 12,  2, 11,  0, 22, 21, 19, 22, 15,  1, 11,  5,  0, 28,  6,  6,
    7, 21, 14, 13,  8, 23, 20, 22,  3, 12, 19, 10, 23, 25,  8, 19, 27,
    0, 21, 11, 15, 21, 15, 19,  9,  0, 11,  3, 13, 25, 14,  4,  0, 21,
    21,  7, 28, 16,  5, 19,  9,  4, 24, 13, 10, 26,  1,  3, 18, 25, 13,
    22,  0, 12,  2, 10,  7, 26, 20, 26,  5, 18,  4, 21,  4, 16, 24, 18,
    21,  7, 13, 25, 14,  7,  6,  2,  6,  7, 14,  5, 10,  0, 19,  9, 12,
    13,  2, 13, 10,  6, 26, 15,  7, 13, 17,  5,  6,  6, 29, 13,  6, 18,
    25, 14, 14,  8,  9, 27, 21, 16,  5,  3,  9,  7, 24, 22, 22,  4, 29,
    0,  6,  3, 22,  1,  7, 29, 23,  5, 29, 21,  0, 16])

# Generated randomly with np.random.randint(1, 20, 200)
weights = np.array([16, 12,  3, 19, 12, 11, 12, 13,  2, 12,  9,  3,  3,  1, 15,  1,  7,
    16, 14, 18,  5, 15, 11,  8, 11,  6,  4, 12,  6,  2, 18, 10, 15, 14,
    9, 16,  7, 13,  7,  4, 18, 15, 11,  1, 17, 14, 14, 10,  3, 11,  8,
    15,  8,  7,  8, 14,  1,  5, 10,  1,  8, 15, 14,  8,  7,  1,  1,  5,
    4, 12,  3, 14,  4, 18,  1,  3, 18, 14,  8,  4,  7,  8, 12,  5,  5,
    4, 14, 18,  7,  3, 14, 16,  9,  8,  1,  4,  5, 13,  3, 14,  9,  3,
    10, 10, 11, 12,  9, 18,  7,  3, 14, 11, 12,  1,  9, 19,  1,  8,  6,
    8,  6, 15,  1,  4, 10,  6,  3, 16,  6,  2, 15, 11,  8, 17, 17, 14,
    5,  4,  6, 12, 14, 12, 19, 18, 11, 15,  1,  9,  2,  8,  7, 14, 10,
    3,  4,  9, 12,  2,  8,  9, 14, 14, 11,  9, 16,  4,  2, 17,  9,  4,
    5,  3,  8,  6,  2, 11,  5,  9, 16,  4, 10,  7,  8, 14,  1,  3, 14,
    14, 16,  7, 19, 16, 10, 10,  7, 10, 12, 10, 19, 16])

# Generated randomly with np.random.randint(1, 10, 200)
limits = np.array([8, 5, 1, 3, 5, 9, 4, 2, 1, 5, 2, 4, 6, 1, 2, 2, 9, 6, 2, 8, 2, 7,
    5, 4, 7, 1, 6, 8, 3, 5, 6, 5, 5, 6, 3, 8, 2, 2, 4, 4, 6, 9, 1, 6,
    7, 8, 2, 6, 6, 8, 2, 8, 3, 8, 6, 5, 1, 7, 3, 6, 8, 9, 9, 3, 9, 2,
    9, 5, 1, 1, 2, 4, 3, 4, 8, 1, 1, 7, 4, 2, 8, 3, 5, 2, 6, 6, 8, 7,
    1, 8, 4, 6, 7, 4, 1, 9, 7, 5, 8, 4, 2, 6, 3, 2, 7, 3, 2, 1, 1, 2,
    9, 8, 1, 4, 3, 8, 1, 9, 7, 5, 2, 7, 4, 8, 3, 1, 5, 5, 7, 6, 9, 6,
    3, 2, 7, 2, 5, 6, 1, 3, 5, 4, 4, 1, 2, 3, 7, 8, 1, 7, 8, 8, 4, 2,
    4, 9, 2, 3, 2, 7, 5, 4, 3, 7, 1, 9, 5, 4, 8, 1, 4, 7, 7, 9, 5, 5,
    4, 7, 1, 9, 4, 6, 5, 3, 6, 8, 2, 4, 7, 3, 3, 6, 7, 9, 2, 6, 4, 4,
    8, 2])