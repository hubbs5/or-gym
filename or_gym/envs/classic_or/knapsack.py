import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils.env_config import assign_env_config
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
    
    def __init__(self, *args, **kwargs):
        # Generate data with consistent random seed to ensure reproducibility
        self.N = 200
        self.item_numbers = np.arange(self.N)
        self.max_weight = 200
        self.current_weight = 0
        self._max_reward = 10000
        self.mask = False
        self.seed = 0
        self.item_weights = weights.copy()
        self.item_values = values.copy()
        # Add env_config, if any
        assign_env_config(self, kwargs)
        self.set_seed()
        
        obs_space = spaces.Box(
            0, self.max_weight, shape=(2*self.N + 1,), dtype=np.int16)
        self.action_space = spaces.Discrete(self.N)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.N,)),
                "avail_actions": spaces.Box(0, 1, shape=(self.N,)),
                "state": obs_space
                })
        else:
            self.observation_space = spaces.Box(
                0, self.max_weight, shape=(2, self.N + 1), dtype=np.int16)
        
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
        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight,
            0, 1)
            state = np.hstack([
                self.item_weights,
                self.item_values,
                np.array([self.current_weight])
                ])
            self.state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.N),
                "state": state
                }
        else:
            state = np.vstack([
                self.item_weights,
                self.item_values])
            self.state = np.hstack([
                state,
                np.array([
                    [self.max_weight],
                     [self.current_weight]])
                ])
        # state = np.hstack([state,
            # np.array([self.current_weight])])
        # self.state = np.hstack([
        #     self.state, 
        #     np.array([[self.max_weight],
        #               [self.current_weight]])
        # ])

       
        # mask = np.where(self.item_limits > 0, mask, 0)
        
    
    def reset(self):
        self.current_weight = 0
        self._update_state()
        return self.state
    
    def sample_action(self):
        return np.random.choice(self.item_numbers)

    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class BinaryKnapsackEnv(KnapsackEnv):
    '''
    Binary Knapsack Problem

    The Binary or 0-1 KP allows selection of each item only once or not at
    all.

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
    def __init__(self, *args, **kwargs):
        self.item_limits_init = np.ones(self.N)
        super().__init__()
        
        self.item_weights = default_bin_weights.copy()
        self.item_weights = default_bin_values.copy()
        assign_env_config(self, kwargs)

        obs_space = spaces.Box(
            0, self.max_weight, shape=(3, self.N + 1), dtype=np.int32)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(len(self.item_limits),)),
                "avail_actions": spaces.Box(0, 1, shape=(len(self.item_limits),)),
                "state": obs_space
            })
        else:
            self.observation_space = obs_space

        self.reset()

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
                self._update_state(item)
            else:
                # End if over weight
                reward = 0
                done = True
        else:
            # End if item is unavailable
            reward = 0
            done = True
            
        return self.state, reward, done, {}

    def _update_state(self, item=None):
        if item is not None:
            self.item_limits[item] -= 1
        state_items = np.vstack([
            self.item_weights,
            self.item_values,
            self.item_limits
        ])
        state = np.hstack([
            state_items, 
            np.array([[self.max_weight],
                      [self.current_weight], 
                      [0] # Serves as place holder
                ])
        ])
        mask = np.where(self.current_weight + self.item_weights > self.max_weight,
            0, 1)
        mask = np.where(self.item_limits > 0, mask, 0)
        self.state = {
            "action_mask": mask,
            "avail_actions": np.ones(self.N),
            "state": state
        }
        
    def sample_action(self):
        return np.random.choice(
            self.item_numbers[np.where(self.item_limits!=0)])
    
    def reset(self):
        self.current_weight = 0
        self.item_limits = self.item_limits_init.copy()
        self._update_state()
        return self.state

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
    def __init__(self, *args, **kwargs):
        self.item_limits_init = default_bkp_limits.copy()
        self.item_limits = self.item_limits_init.copy()
        super().__init__()
        self.item_weights = default_bkp_weights.copy()
        self.item_values = default_bkp_values.copy()

        assign_env_config(self, kwargs)
        
        obs_space = spaces.Box(
            0, self.max_weight, shape=(3, self.N + 1), dtype=np.int32)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(len(self.item_limits),)),
                "avail_actions": spaces.Box(0, 1, shape=(len(self.item_limits),)),
                "state": obs_space
            })
        else:
            self.observation_space = obs_space

        self._max_reward = 1800 # Used for VF clipping
        
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
                self._update_state(item)
            else:
                # End if over weight
                reward = 0
                done = True
        else:
            # End if item is unavailable
            reward = 0
            done = True
            
        return self.state, reward, done, {}

    def _update_state(self, item=None):
        if item is not None:
            self.item_limits[item] -= 1
        state_items = np.vstack([
            self.item_weights,
            self.item_values,
            self.item_limits
        ])
        state = np.hstack([
            state_items, 
            np.array([[self.max_weight],
                      [self.current_weight], 
                      [0] # Serves as place holder
                ])
        ])
        mask = np.where(self.current_weight + self.item_weights > self.max_weight,
            0, 1)
        mask = np.where(self.item_limits > 0, mask, 0)
        self.state = {
            "action_mask": mask,
            "avail_actions": np.ones(self.N),
            "state": state
        }
        
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
    def __init__(self, *args, **kwargs):
        BoundedKnapsackEnv.__init__(self)
        self.item_values = default_okp_values.copy()
        self.item_weights = default_okp_weights.copy()
        assign_env_config(self, kwargs)
        self.action_space = spaces.Discrete(2)

        obs_space = spaces.Box(0, self.max_weight, shape=(4,))
        if self.mask:
            self.observation_space = spaces.Dict({
                'state': obs_space,
                'avail_actions': spaces.Box(0, 1, shape=(2,)),
                'action_mask': spaces.Box(0, 1, shape=(2,))
            })
        else:
            self.observation_space = obs_space

        self.step_counter = 0
        self.step_limit = 50
        
        self.state = self.reset()
        self._max_reward = 600
        
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
                self._update_state()
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
            
        return self.state, reward, done, {}
    
    def _update_state(self):
        self.current_item = np.random.choice(self.item_numbers, p=self.item_probs)
        self.state = (
            # self.item_weights,
            # self.item_values,
            # self.item_probs,
            np.array([
                # self.max_weight,
                self.current_weight,
                self.current_item,
                self.item_weights[self.current_item],
                self.item_values[self.current_item]
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

# Default parameters
default_bin_weights = np.array([59, 76, 78, 48, 58, 43, 49, 95, 31, 62, 12,
    24, 21, 47,  2, 28,  8, 16, 75, 68, 65,  5, 91,  8, 39, 64, 87, 26, 99,
    26, 84, 45, 52, 43, 54, 92, 1, 4, 83, 54, 38, 25, 28,  5, 52, 77, 32, 
    48, 63, 53, 57, 59, 33, 55, 91, 43, 42, 36, 35,  6, 58, 92, 35, 62, 44,
    26, 18,  7, 49, 84, 55, 23, 14, 11, 44, 51, 81, 31, 40, 60, 61, 21, 45,
    98, 47, 23,  3, 41, 78, 48, 96, 83, 36, 45, 23, 24, 54, 23, 82, 97, 53,
    68, 39, 29, 61, 30,  8, 64, 94, 36, 29, 54, 91, 99, 21, 78,  1, 19, 71,
    64, 52, 91, 35, 25, 97, 41, 13, 52, 82,  5, 28, 40, 96,  3, 48, 26, 46, 
    43, 99, 80, 40, 34, 35, 92, 61, 17, 77, 62, 57, 71, 74, 36, 70, 21, 53, 
    51, 69, 62, 48, 64, 74, 29, 20,  1, 31, 69, 82, 73, 77, 61, 69, 52, 95,
    94, 59, 76, 48, 29, 38, 71, 66, 50,  8, 27, 33, 44, 18, 52, 59, 93, 32,
    89, 65, 92, 77, 23, 47, 16, 19, 89])

default_bin_values = np.array([18, 64, 79, 30, 79, 33, 69, 87, 17, 56, 95, 5, 
    88, 39, 18, 80, 59, 9, 71, 80, 85, 44, 78, 88,  2, 40, 96, 99, 18, 
    99, 56, 64, 49, 90, 18, 52, 44, 57, 60, 33,  3, 22, 14, 74, 11,  2,
    23,  8, 90, 17, 44, 0, 29, 80, 50,  9, 28, 19, 15, 32, 82, 98, 91, 
    41, 73, 35, 69,  3, 80,  0, 35, 97, 26, 66, 47,  0, 21, 27, 66, 91,
    54, 73, 79, 86, 63, 3, 35, 60, 68, 33, 84, 40, 84, 46, 39, 15, 21, 
    98, 93, 92, 46, 49, 26, 90, 44, 89, 34, 13, 52, 56, 24, 18, 77, 46,
    98, 67, 66, 74, 40, 5, 44, 49, 61, 54, 89, 87, 75, 58, 51, 22, 41, 
    54, 50, 93,  1, 17, 93, 21, 52, 99, 94, 57, 90, 49, 58, 41, 16, 54,
    60, 70, 71, 34, 37, 43, 77, 89, 20, 40, 82, 17, 26, 92, 72, 44, 37, 
    2, 14, 45, 46, 31, 18, 73, 28, 62, 33, 40, 18, 72, 13, 26, 44, 69, 
    47, 71, 65, 54, 71, 92, 29, 96, 71, 72, 51, 82, 95, 87, 73, 79, 37, 
    51])

default_ukp_values = None
default_ukp_weights = None

default_bkp_values = None
default_bkp_weights = None
default_bkp_limits = None

default_okp_values = np.array([ 4, 13,  5, 25, 11, 15, 13, 17, 21, 18, 27,  2,  4, 13,  0,  0,  0,
    15,  6, 29, 26, 11, 27, 26, 16, 25, 19, 19, 22,  3, 22, 28, 15,  9,
    9, 12,  0, 15, 25,  8, 23, 23, 17,  8, 24,  1, 11,  4, 12, 24, 27,
    20,  1, 27,  8,  5, 19, 14, 13, 11, 21, 23,  5, 27, 24,  8, 29,  3,
    2, 22, 22,  1,  5, 21, 29,  8,  2,  4,  0,  9, 22,  0,  2,  3, 14,
    18,  8,  4,  7, 17, 19, 22, 21,  9, 18, 25, 28, 15, 23,  6, 15, 16,
    22, 14, 15,  3,  3, 16, 15,  2,  0,  6,  9,  9, 22, 20, 18,  7, 23,
    6, 22, 16, 13, 16,  5, 17, 23,  2, 17, 18, 21, 16, 17, 14, 25, 17,
    19,  9, 12, 23, 10,  6, 13, 17,  8, 22, 12, 12, 22, 16,  9, 16,  2,
    19,  3, 28, 15, 23, 17,  4, 27, 23, 25,  9, 28,  8,  5,  5,  1, 11,
    6,  5,  9, 11, 28, 26, 28, 25, 18, 23, 20, 29,  5, 13, 26, 28, 19,
    29, 28, 17, 21, 18,  4, 27, 21, 22, 10,  3,  8, 16])

default_okp_weights = np.array([17,  9,  6,  1,  7,  9,  1,  1, 13, 14,  6,  1, 10, 19, 15, 15, 13,
       11,  1,  9,  8,  2,  6,  6,  2, 18,  1, 12, 12,  8, 16, 17,  2, 10,
        3,  2, 13,  6,  3,  5,  7, 10,  6,  1, 12, 12, 13, 12,  1, 19,  3,
        8, 18, 19, 19, 18,  1,  7, 14, 15,  7, 17,  1,  5,  1, 16,  3, 15,
       11, 10,  6, 13, 11,  3, 19, 13, 11,  8, 16, 10, 15, 13,  8, 12, 13,
       19, 11,  8,  9,  3,  4,  8,  8,  7,  8,  8, 12,  4,  7, 16, 17, 18,
        2, 14, 14,  1, 18, 13, 18,  6, 16,  1,  7, 19, 15, 15, 10,  9,  2,
        4, 17,  3,  3, 14, 13,  3, 14,  7,  4,  2,  3,  2,  6, 13, 11,  9,
        1, 11,  9,  5, 15, 12, 12,  2,  9, 14,  5,  3, 14, 16, 18, 17, 11,
       16, 17, 10, 11,  1, 11, 15, 11, 16, 14,  5,  5,  7, 10,  9,  7, 17,
        4,  9, 14, 18,  8, 18,  3,  6, 19,  9,  5, 10, 13,  8, 12,  3, 12,
        5,  1,  2,  9, 13,  3, 18,  8, 13,  4,  3,  7, 14])

# limits = np.array([3, 3, 8, 9, 8, 8, 3, 7, 8, 6, 4, 9, 8, 8, 5, 3, 9, 7, 8, 2, 3, 4,
#        2, 8, 6, 9, 7, 2, 2, 2, 8, 6, 2, 5, 3, 5, 7, 2, 3, 1, 4, 6, 4, 7,
#        8, 3, 8, 7, 3, 6, 2, 7, 9, 3, 3, 8, 3, 9, 8, 7, 2, 7, 1, 7, 4, 5,
#        1, 3, 4, 2, 1, 4, 4, 3, 9, 4, 3, 9, 6, 5, 2, 5, 2, 2, 7, 4, 5, 1,
#        8, 5, 2, 5, 4, 3, 9, 5, 9, 1, 8, 8, 3, 2, 1, 9, 5, 4, 6, 8, 7, 2,
#        6, 8, 9, 3, 1, 9, 6, 3, 2, 4, 9, 2, 2, 4, 5, 8, 3, 8, 9, 3, 1, 2,
#        6, 1, 5, 6, 7, 3, 9, 4, 4, 4, 2, 9, 8, 6, 5, 9, 8, 4, 5, 5, 3, 5,
#        7, 2, 8, 6, 7, 5, 2, 7, 5, 2, 3, 7, 9, 2, 1, 4, 8, 8, 9, 8, 8, 7,
#        7, 3, 2, 7, 3, 5, 4, 9, 1, 8, 1, 5, 7, 4, 2, 8, 8, 6, 1, 7, 2, 2,
#        5, 1])
