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
    def __init__(self, *args, **kwargs):
        # Normalized Capacities
        self.cpu_capacity = 1
        self.ram_capacity = 1
        self.step_limit = 60 * 24 / 15
        self.n_pms = 100 # Number of physical machines to choose from
        self.load_idx = np.array([1, 2]) # Gives indices for CPU and mem reqs
        # Add env_config, if any
        for key, value in kwargs.items():
            setattr(self, key, value)
        if hasattr(self, 'env_config'):
            for key, value in self.env_config.items():
                setattr(self, key, value)

        self.observation_space = spaces.Tuple((
            spaces.Box(
                low=-0.1, high=1, shape=(self.n_pms, 3), dtype=np.float32), # Imprecision causes some errors
            spaces.Box(
                low=0, high=1, shape=(2,), dtype=np.float32)
            ))
        self.action_space = spaces.Discrete(self.n_pms)        
        self.state = self.reset()
        
    def step(self, action):
        done = False
        pm_state = self.state[0] # Physical machine state
        if action < 0 or action >= self.n_pms:
            raise ValueError('Invalid Action')
        elif any(pm_state[action, 1:] + self.demand[self.step_count] > 1):
            # Demand doesn't fit into PM
            reward = -100
            done = True
        else:
            if pm_state[action, 0] == 0:
                # Open PM if closed
                pm_state[action, 0] = 1
            pm_state[action, self.load_idx] += self.demand[self.step_count]
            reward = np.sum(pm_state[:, 0] * 
                (pm_state[:, 1] - 1 + pm_state[:, 2] - 1))
            
        self.step_count += 1
        if self.step_count >= self.step_limit:
            done = True
            reward = 0
        else:
            self.state = (pm_state, self.demand[self.step_count])
        
        return self.state, reward, done, {}
    
    def reset(self):
        self.step_count = 0
        self.assignments = {}
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
    def __init__(self, *args, **kwargs):
        super().__init__()       
        self.state = self.reset()
        
    def step(self, action):
        done = False
        pm_state = self.state[0]
        if action < 0 or action >= self.n_pms:
            raise ValueError('Invalid Action')
        elif any(pm_state[action, 1:] + self.demand[self.step_count] > 1):
            # Demand doesn't fit into PM
            reward = -100
            done = True
        else:
            if pm_state[action, 0] == 0:
                # Open PM if closed
                pm_state[action, 0] = 1
            pm_state[action, self.load_idx] += self.demand[self.step_count]
            self.assignments[self.step_count] = action
        
        # Remove processes
        if self.step_count in self.durations.values():
            for process in self.durations.keys():
                # Remove process from PM
                if self.durations[process] == self.step_count:
                    pm = self.assignments[process] # Find PM where process was assigned
                    pm_state[pm, self.load_idx] -= self.demand[process]
                    # Shut down PM's if state is 0
                    if pm_state[pm, self.load_idx].sum() == 0:
                        pm_state[pm, 0] = 0
            
        if self.step_count >= self.step_limit:
            done = True
            reward = 0
        else:
            self.state = (pm_state, self.demand[self.step_count])
            
        if not done:
            reward = np.sum(pm_state[:, 0] * 
                (pm_state[:, 1] - 1 + pm_state[:, 2] - 1))
        
        self.step_count += 1
        
        return self.state, reward, done, {}
        
    def reset(self):
        self.step_count = 0
        self.assignments = {}
        self.demand = generate_demand()
        self.durations = generate_durations(self.demand)
        self.state = (np.zeros((self.n_pms, 3)), self.demand[0])
        return self.state

def generate_demand():
    t_int = 5
    n = int(1440 / t_int) # 1 day, 5 minute intervals
    # From Azure data
    mem_probs = np.array([0.12 , 0.165, 0.328, 0.287, 0.064, 0.036])
    mem_bins = np.array([0.02857143, 0.05714286, 0.11428571, 0.45714286, 0.91428571,
       1.]) # Normalized bin sizes
    mu_cpu = 16.08
    sigma_cpu = 1.26
    cpu_demand = np.random.normal(loc=mu_cpu, scale=sigma_cpu, size=n)
    cpu_demand = np.where(cpu_demand<=0, mu_cpu, cpu_demand) # Ensure demand isn't negative
    mem_demand = np.random.choice(mem_bins, p=mem_probs, size=n)
    return np.vstack([cpu_demand/100, mem_demand]).T

def generate_durations(demand):
    # duration_params = np.array([ 6.53563303e-02,  5.16222242e+01,  4.05028032e+06, -4.04960880e+06])
    return {i: np.random.randint(low=i+1, high=len(demand)+1)
        for i, j in enumerate(demand)}

def gaussian_model(params, x):
    return params[2] * np.exp(-0.5*((x - params[0]) / params[1]) ** 2) + params[3]