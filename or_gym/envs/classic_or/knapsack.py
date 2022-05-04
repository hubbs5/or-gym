import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils import assign_env_config
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
    
    # Internal list of placed items for better rendering
    _collected_items = []
    
    def __init__(self, *args, **kwargs):
        # Generate data with consistent random seed to ensure reproducibility
        self.N = 200
        self.max_weight = 200
        self.current_weight = 0
        self._max_reward = 10000
        self.mask = True
        self.seed = 0
        self.item_numbers = np.arange(self.N)
        self.item_weights = np.random.randint(1, 100, size=self.N)
        self.item_values = np.random.randint(0, 100, size=self.N)
        self.over_packed_penalty = 0
        self.randomize_params_on_reset = False
        self._collected_items.clear()
        # Add env_config, if any
        assign_env_config(self, kwargs)
        self.set_seed()

        obs_space = spaces.Box(
            0, self.max_weight, shape=(2*self.N + 1,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.N)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.N,), dtype=np.int32),
                "avail_actions": spaces.Box(0, 1, shape=(self.N,), dtype=np.int16),
                "state": obs_space
                })
        else:
            self.observation_space = spaces.Box(
                0, self.max_weight, shape=(2, self.N + 1), dtype=np.int32)
        
        self.reset()
        
    def _STEP(self, item):
        # Check that item will fit
        if self.item_weights[item] + self.current_weight <= self.max_weight:
            self.current_weight += self.item_weights[item]
            reward = self.item_values[item]
            self._collected_items.append(item)
            if self.current_weight == self.max_weight:
                done = True
            else:
                done = False
        else:
            # End trial if over weight
            reward = self.over_packed_penalty
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
                "avail_actions": np.ones(self.N, dtype=np.int16),
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
    
    def _RESET(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
        self.current_weight = 0
        self._collected_items.clear()
        self._update_state()
        return self.state
    
    def sample_action(self):
        return np.random.choice(self.item_numbers)

    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        return self._RESET()

    def step(self, action):
        return self._STEP(action)
        
    def render(self):
        total_value = 0
        total_weight = 0
        for i in range(self.N) :
            if i in self._collected_items :
                total_value += self.item_values[i]
                total_weight += self.item_weights[i]
        print(self._collected_items, total_value, total_weight)
        
        # RlLib requirement: Make sure you either return a uint8/w x h x 3 (RGB) image or handle rendering in a window and then return `True`.
        return True

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
        super().__init__()
        self.item_weights = np.random.randint(1, 100, size=self.N)
        self.item_values = np.random.randint(0, 100, size=self.N)
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

    def _STEP(self, item):
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
        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight,
                0, 1)
            mask = np.where(self.item_limits > 0, mask, 0)
            self.state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.N),
                "state": state
            }
        else:
            self.state = state.copy()
        
    def sample_action(self):
        return np.random.choice(
            self.item_numbers[np.where(self.item_limits!=0)])
    
    def _RESET(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
        self.current_weight = 0
        self.item_limits = np.ones(self.N)
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
        self.N = 200
        self.item_limits_init = np.random.randint(1, 10, size=self.N)
        self.item_limits = self.item_limits_init.copy()
        super().__init__()
        self.item_weights = np.random.randint(1, 100, size=self.N)
        self.item_values = np.random.randint(0, 100, size=self.N)

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
        
    def _STEP(self, item):
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
        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight,
                0, 1)
            mask = np.where(self.item_limits > 0, mask, 0)
            self.state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.N),
                "state": state
            }
        else:
            self.state = state.copy()
        
    def sample_action(self):
        return np.random.choice(
            self.item_numbers[np.where(self.item_limits!=0)])
    
    def _RESET(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
            self.item_limits = np.random.randint(1, 10, size=self.N)
        else:
            self.item_limits = self.item_limits_init.copy()

        self.current_weight = 0
        self._update_state()
        return self.state

class OnlineKnapsackEnv(BoundedKnapsackEnv):
    '''
    Online Knapsack Problem

    The Knapsack Problem (KP) is a combinatorial optimization problem which
    requires the user to select from a range of goods of different values and
    weights in order to maximize the value of the selected items within a 
    given weight limit. This version is online meaning each item is randomly
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
        
    def _STEP(self, action):
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
        
        self._update_state()
        self.step_counter += 1
        if self.step_counter >= self.step_limit:
            done = True
            
        return self.state, reward, done, {}
    
    def _update_state(self):
        self.current_item = np.random.choice(self.item_numbers, p=self.item_probs)
        current_item_weight = self.item_weights[self.current_item]
        state = np.array([
                self.current_weight,
                self.current_item,
                current_item_weight,
                self.item_values[self.current_item]
                ])
        if self.mask:
            mask = np.ones(2)
            if current_item_weight + self.current_weight > self.max_weight:
                mask[1] = 0
            self.state = {
                'state': state,
                'avail_actions': np.ones(2),
                'action_mask': mask
            }            
        else:
            self.state = state

    def sample_action(self):
        return np.random.choice([0, 1])
    
    def _RESET(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
            self.item_limits = np.random.randint(1, 10, size=self.N)
        else:
            self.item_limits = self.item_limits_init.copy()

        if not hasattr(self, 'item_probs'):
            self.item_probs = self.item_limits_init / self.item_limits_init.sum()
        self.current_weight = 0
        self.step_counter = 0
        self._update_state()
        return self.state