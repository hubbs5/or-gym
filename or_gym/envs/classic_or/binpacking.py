'''
Example taken from Balaji et al.
Paper: https://arxiv.org/abs/1911.10641
GitHub: https://github.com/awslabs/or-rl-benchmarks
'''
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils import assign_env_config
import copy

BIG_NEG_REWARD = -100
BIG_POS_REWARD = 10

class BinPackingEnv(gym.Env):
    '''
    Small Bin Packing with Bounded Waste
    Env Registration: BinPacking-v0

    The Bin Packing Problem (BPP) is a combinatorial optimization problem which
    requires the user to select from a range of goods of different values and
    weights in order to maximize the value of the selected items within a 
    given weight limit. This version is online meaning each item is randonly
    presented to the algorithm one at a time, at which point the algorithm 
    can either accept or reject the item. After seeing a fixed number of 
    items are shown, the episode terminates. If the weight limit is reached
    before the episode ends, then it terminates early.

    Observation:
        If mask == False:
            Type: Discrete
            0 - bin_capacity: Count of bins at a given level h
            -1: Current item size
        if mask == True:
            Type: Dict
            'state': vector of bins where 0 to bin capacity is the count of
                bins at that load level h and the last entry is the current
                item size.
            'action_mask': binary vector where 0 indicates infeasible
                actions and 1 feasible actions.
            'avail_actions': vector of values to be combined with mask.

    Actions:
        Type: Discrete
        0: Open a new bin and place item into bin
        1+: Attempt to place item into bin at the corresponding level

    Reward:
        Negative of the waste, which is the difference between the current
        size and excess space of the bin.

    Starting State:
        No available bins and random starting item
        
    Episode Termination:
        When invalid action is selected (e.g. attempt to place item in non-existent
        bin), bin limits are exceeded, or step limit is reached.
    '''
    def __init__(self, *args, **kwargs):
        self.bin_capacity = 9
        self.item_sizes = [2, 3]
        self.item_probs = [0.8, 0.2]
        self.step_count = 0
        self.step_limit = 100
        self.mask = False
        assign_env_config(self, kwargs)
        self._build_obs_space()
        self._check_settings()
        self.seed()
        self.state = self.reset()
        
    def _STEP(self, action):
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
        
        self.step_count += 1 
        
        if self.step_count >= self.step_limit:
            done = True
        
        self.state = self._update_state()
        
        return self.state, reward, done, {}

    def _update_state(self):
        self.item_size = self.get_item()
        state = np.array(self.bin_levels + [self.item_size])
        if self.mask:
            state_dict = {
                'state': state,
                'avail_actions': np.ones(self.bin_capacity)}
            # Mask actions for closed bins
            mask = np.ones(self.bin_capacity) * np.array(state[:-1])
            # Mask actions where packing would exceed capacity
            overflow = self.bin_capacity - self.item_size
            mask[overflow+1:] = 0
            # Ensure open new bin is available
            mask[0] = 1
            state_dict['action_mask'] = mask
            return state_dict
        else:
            return state
    
    def get_item(self):
        return np.random.choice(self.item_sizes, p=self.item_probs)
        
    def sample_action(self):
        return self.action_space.sample()
    
    def _RESET(self):
        self.current_weight = 0
        self.step_count = 0        
        self.num_full_bins = 0
        self.total_reward = 0
        self.waste = 0
        self.bin_levels = [0] * self.bin_capacity
        self.item_size = self.get_item()
        self.state = self._update_state()
        return self.state

    def _build_obs_space(self):
        if self.mask:
            self.observation_space = spaces.Dict({
                'action_mask': spaces.Box(0, 1, 
                    shape=(self.bin_capacity,),
                    dtype=np.uint8),
                'avail_actions': spaces.Box(0, 1, 
                    shape=(self.bin_capacity,),
                    dtype=np.uint8),
                'state': spaces.Box(
                    low=np.array([0] * (1 + self.bin_capacity)),
                    high=np.array([self.step_limit] * self.bin_capacity +
                        [max(self.item_sizes)]),
                    dtype=np.uint32)
            })
        else:
            self.observation_space = spaces.Box(
                low=np.array([0] * (1 + self.bin_capacity)),
                high=np.array([self.step_limit] * self.bin_capacity + 
                    [max(self.item_sizes)]),
                dtype=np.uint32)
        
        self.action_space = spaces.Discrete(self.bin_capacity)

    def _check_settings(self):
        # Ensure setting sizes and probs are correct at initialization
        assert sum(self.item_probs) == 1, 'Item probabilities do not sum to 1.'
        assert len(self.item_probs) == len(self.item_sizes), \
            'Dimension mismatch between item probabilities' + \
                ' ({}) and sizes ({})'.format(
                len(self.item_probs), len(self.item_sizes))

    def reset(self):
        return self._RESET()

    def step(self, action):
        return self._STEP(action)

class BinPackingLW1(BinPackingEnv):
    '''
    Large Bin Packing Probem with Bounded Waste
    Env Registration: BinPacking-v1
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bin_capacity = 100
        self.item_probs = [0.14, 0.1, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19]
        self.item_sizes = np.arange(1, 10)
        self.step_limit = 1000
        assign_env_config(self, kwargs)
        self._build_obs_space()
        self._check_settings()
        self.seed()
        self.state = self.reset()

class BinPackingPP0(BinPackingEnv):
    '''
    Small Perfectly Packable Bin Packing with Linear Waste
    Env Registration: BinPacking-v2
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.item_probs = [0.75, 0.25]
        assign_env_config(self, kwargs)
        self._build_obs_space()
        self._check_settings()
        self.seed()
        self.state = self.reset()

class BinPackingPP1(BinPackingPP0):
    '''
    Large Bin Packing Probem with Bounded Waste
    Env Registration: BinPacking-v3
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bin_capacity = 100
        self.item_probs = [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33]
        self.item_sizes = np.arange(1, 10)
        self.step_limit = 1000
        assign_env_config(self, kwargs)
        self._build_obs_space()
        self._check_settings()
        self.seed()
        self.state = self.reset()

class BinPackingBW0(BinPackingEnv):
    '''
    Small Perfectly Packable Bin Packing Problem with Bounded Waste
    Env Registration: BinPacking-v4
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.item_probs = [0.5, 0.5]
        assign_env_config(self, kwargs)
        self._build_obs_space()
        self._check_settings()
        self.seed()
        self.state = self.reset()

class BinPackingBW1(BinPackingBW0):
    '''
    Large Perfectly Packable Bin Packing Problem with Bounded Waste
    Env Registration: BinPacking-v5
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bin_capacity = 100
        self.item_probs = [0, 0, 0, 1/3, 0, 0, 0, 0, 2/3]
        self.item_sizes = np.arange(1, 10)
        self.step_limit = 1000
        assign_env_config(self, kwargs)
        self._build_obs_space()
        self._check_settings()
        self.seed()
        self.state = self.reset()