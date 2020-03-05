import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
import copy

class VMPackingEnv(gym.Env):
	'''
    Online VM Packing Problem

    The VM Packing Problem (VMPP) is a combinatorial optimization problem which
    requires the user to select from a series of physical machines (PM's) to
    send a virtual machine process to. Each VM process is characterized by
    two values, the memory and compute of the process. These are normalized
    by the PM capacities to range between 0-1. 

    Observation:
        Type: Tuple, Discrete
        [0][:, 0]: Binary indicator for open PM's
        [0][:, 1]: CPU load of PM's
        [0][:, 2]: Memory load of PM's
        [1][0]: Current CPU demand
        [1][1]: Current memory demand

    Actions:
        Type: Discrete
        Integer of PM number to send VM to that PM

    Reward:
        Negative of the waste, which is the difference between the current
        size and excess space on the PM.

    Starting State:
        No open PM's and random starting item
        
    Episode Termination:
        When invalid action is selected, attempt to overload VM, or step
        limit is reached.
    '''
    def __init__(self):
        # Normalized Capacities
        self.cpu_capacity = 1
        self.ram_capacity = 1
        self.step_limit = 60 * 24 / 15
        self.n_pms = 100 # Number of physical machines to choose from
        
        self.observation_space = spaces.Box(
            low=np.zeros((self.n_pms, 3), dtype=np.float32),
            high=np.ones((self.n_pms, 3), dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_pms)
        
        self.state = self.reset()
        
    def step(self, action):
        done = False
        pm_state = self.state[0]
        if action < 0 or action >= self.n_pms:
            raise ValueError('Invalid Action')
        elif any(pm_state[action, 0] + self.demand[self.step_count] > 1):
            # Demand doesn't fit into PM
            reward = -100
            done = True
        else:
            if pm_state[action, 0] == 0:
                # Open PM if closed
                pm_state[action, 0] = 1
            pm_state[action, np.array([1, 2])] += self.demand[self.step_count]
            reward = np.sum(pm_state[:, 0] * 
                (pm_state[:, 1] - 1 + pm_state[:, 2] - 1))
            
        self.step_count += 1
        if self.step_count >= self.step_limit:
            done = True
            reward = 0
        self.state = (pm_state, self.demand[self.step_count])
        
        return self.state, reward, done, {}
    
    def reset(self):
        self.step_count = 0
        self.demand = generate_demand()
        self.state = (np.zeros((self.n_pms, 3)), self.demand[0])
        return self.state

class TempVMPackingEnv(VMPackingEnv):
    '''
    Online Temporary VM Packing Problem

    The VM Packing Problem (VMPP) is a combinatorial optimization problem which
    requires the user to select from a series of physical machines (PM's) to
    send a virtual machine process to. Each VM process is characterized by
    two values, the memory and compute of the process. These are normalized
    by the PM capacities to range between 0-1. 

    Observation:
        Type: Tuple, Discrete
        [0][:, 0]: Binary indicator for open PM's
        [0][:, 1]: CPU load of PM's
        [0][:, 2]: Memory load of PM's
        [1][0]: Current CPU demand
        [1][1]: Current memory demand

    Actions:
        Type: Discrete
        Integer of PM number to send VM to that PM

    Reward:
        Negative of the waste, which is the difference between the current
        size and excess space on the PM.

    Starting State:
        No open PM's and random starting item
        
    Episode Termination:
        When invalid action is selected, attempt to overload VM, or step
        limit is reached.
    '''
    def __init__(self):
        super().__init__()       
        self.state = self.reset()
        
    def step(self, action):
        done = False
        pm_state = self.state[0]
        if action < 0 or action >= self.n_pms:
            raise ValueError('Invalid Action')
        elif any(pm_state[action, 0] + self.demand[self.step_count] > 1):
            # Demand doesn't fit into PM
            reward = -100
            done = True
        else:
            if pm_state[action, 0] == 0:
                # Open PM if closed
                pm_state[action, 0] = 1
            pm_state[action, self.loads] += self.demand[self.step_count]
            self.assignments[self.step_count] = action
        
        # Remove processes
        if self.step_count in self.durations.values():
            for process in self.durations.keys():
                # Remove process from PM
                if self.durations[process] == self.step_count:
                    pm = alist[process]
                    pm_state[pm, self.loads] -= env.demand[process]
                    # Shut down PM's if state is 0
                    if pm_state[pm, self.loads].sum() == 0:
                        pm_state[pm, 0] = 0
            
        if self.step_count >= self.step_limit:
            done = True
            reward = 0
            
        if not done:
            reward = np.sum(pm_state[:, 0] * 
                (pm_state[:, 1] - 1 + pm_state[:, 2] - 1))
        
        self.state = (pm_state, self.demand[self.step_count])
        self.step_count += 1
        
        return self.state, reward, done, {}
        
    def reset(self):
        self.step_count = 0
        self.assignments = {}
        self.demand = generate_demand()
        self.durations = generate_durations(self.demand)
        self.state = (np.zeros((self.n_pms, 3)), self.demand[0])
        return self.state

# Placeholder demand generation function
def generate_demand():
    t_int = 15
    n_steps = int(1440 / t_int) # 1 day, 15 minute intervals
    steps = np.arange(n_steps)
    level = np.abs(np.sin(steps) + 5)
    noise = np.random.normal(size=n_steps)
    trend = np.sin(steps / n_steps * 2*np.pi + np.pi)
    cpu_demand = level + noise + trend
    cpu_demand /= cpu_demand.max()
    
    ram_levels = np.array([2, 4, 8, 16, 32, 64, 128])
    # Assume levels are poisson distributed around 3 with each 
    # value mapping to one of the levels
    ram_sample = np.random.poisson(lam=3, size=n_steps)
    ram_demand = np.array([ram_levels[i] 
        if i < len(ram_levels) else max(ram_levels) 
        for i in ram_sample]) / max(ram_levels)
    return np.vstack([cpu_demand, ram_demand]).T

def generate_durations(demand):
    return {i: np.random.randint(low=i+1, high=len(demand)+1)
        for i, j in enumerate(demand)}