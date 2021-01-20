import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils import assign_env_config
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
        self.cpu_capacity = 1
        self.mem_capacity = 1
        self.t_interval = 20
        self.tol = 1e-5
        self.step_limit = int(60 * 24 / self.t_interval)
        self.n_pms = 50
        self.load_idx = np.array([1, 2])
        self.seed = 0
        self.mask = True
        assign_env_config(self, kwargs)
        self.action_space = spaces.Discrete(self.n_pms)

        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.n_pms,)),
                "avail_actions": spaces.Box(0, 1, shape=(self.n_pms,)),
                "state": spaces.Box(0, 1, shape=(self.n_pms+1, 3))
            })
        else:
            self.observation_space = spaces.Box(0, 1, shape=(self.n_pms+1, 3))
        self.reset()
        
    def _RESET(self):
        self.demand = self.generate_demand()
        self.current_step = 0
        self.state = {
            "action_mask": np.ones(self.n_pms),
            "avail_actions": np.ones(self.n_pms),
            "state": np.vstack([
                np.zeros((self.n_pms, 3)),
                self.demand[self.current_step]])
        }
        self.assignment = {}
        return self.state
    
    def _STEP(self, action):
        done = False
        pm_state = self.state["state"][:-1]
        demand = self.state["state"][-1, 1:]
        
        if action < 0 or action >= self.n_pms:
            raise ValueError("Invalid action: {}".format(action))
            
        elif any(pm_state[action, 1:] + demand > 1 + self.tol):
            # Demand doesn't fit into PM
            reward = -1000
            done = True
        else:
            if pm_state[action, 0] == 0:
                # Open PM if closed
                pm_state[action, 0] = 1
            pm_state[action, self.load_idx] += demand
            reward = np.sum(pm_state[:, 0] * (pm_state[:,1:].sum(axis=1) - 2))
            self.assignment[self.current_step] = action
            
        self.current_step += 1
        if self.current_step >= self.step_limit:
            done = True
        self.update_state(pm_state)
        return self.state, reward, done, {}
    
    def update_state(self, pm_state):
        # Make action selection impossible if the PM would exceed capacity
        step = self.current_step if self.current_step < self.step_limit else self.step_limit-1
        data_center = np.vstack([pm_state, self.demand[step]])
        data_center = np.where(data_center>1,1,data_center) # Fix rounding errors
        self.state["state"] = data_center
        self.state["action_mask"] = np.ones(self.n_pms)
        self.state["avail_actions"] = np.ones(self.n_pms)
        if self.mask:
            action_mask = (pm_state[:, 1:] + self.demand[step, 1:]) <= 1
            self.state["action_mask"] = (action_mask.sum(axis=1)==2).astype(int)

    def sample_action(self):
        return self.action_space.sample()

    def generate_demand(self):
        n = self.step_limit
        # From Azure data
        mem_probs = np.array([0.12 , 0.165, 0.328, 0.287, 0.064, 0.036])
        mem_bins = np.array([0.02857143, 0.05714286, 0.11428571, 0.45714286, 0.91428571,
           1.]) # Normalized bin sizes
        mu_cpu = 16.08
        sigma_cpu = 1.26
        cpu_demand = np.random.normal(loc=mu_cpu, scale=sigma_cpu, size=n)
        cpu_demand = np.where(cpu_demand<=0, mu_cpu, cpu_demand) # Ensure demand isn't negative
        mem_demand = np.random.choice(mem_bins, p=mem_probs, size=n)
        return np.vstack([np.arange(n)/n, cpu_demand/100, mem_demand]).T

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()

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
        pm_state = self.state["state"][:-1]
        demand = self.state["state"][-1, 1:]
        
        if action < 0 or action >= self.n_pms:
            raise ValueError("Invalid action: {}".format(action))
            
        elif any(pm_state[action, 1:] + demand > 1 + self.tol):
            # Demand doesn't fit into PM
            reward = -1000
            done = True
        else:
            if pm_state[action, 0] == 0:
                # Open PM if closed
                pm_state[action, 0] = 1
            pm_state[action, self.load_idx] += demand
            reward = np.sum(pm_state[:, 0] * (pm_state[:,1:].sum(axis=1) - 2))
            self.assignment[self.current_step] = action

        # Remove processes
        if self.current_step in self.durations.values():
            for process in self.durations.keys():
                # Remove process from PM
                if self.durations[process] == self.current_step:
                    pm = self.assignment[process] # Find PM where process was assigned
                    pm_state[pm, self.load_idx] -= self.demand[process]
                    # Shut down PM's if state is 0
                    if pm_state[pm, self.load_idx].sum() == 0:
                        pm_state[pm, 0] = 0
            
        self.current_step += 1
        if self.current_step >= self.step_limit:
            done = True
        self.update_state(pm_state)
        return self.state, reward, done, {}
    
    def update_state(self, pm_state):
        # Make action selection impossible if the PM would exceed capacity
        step = self.current_step if self.current_step < self.step_limit else self.step_limit-1
        data_center = np.vstack([pm_state, self.demand[step]])
        data_center = np.where(data_center>1,1,data_center) # Fix rounding errors
        self.state["state"] = data_center
        self.state["action_mask"] = np.ones(self.n_pms)
        self.state["avail_actions"] = np.ones(self.n_pms)
        if self.mask:
            action_mask = (pm_state[:, 1:] + self.demand[step, 1:]) <= 1
            self.state["action_mask"] = (action_mask.sum(axis=1)==2).astype(int)
        
    def _RESET(self):
        self.current_step = 0
        self.assignment = {}
        self.demand = self.generate_demand()
        self.durations = generate_durations(self.demand)
        self.state = (np.zeros((self.n_pms, 3)), self.demand[0])
        return self.state

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()

def generate_durations(demand):
    # duration_params = np.array([ 6.53563303e-02,  5.16222242e+01,  4.05028032e+06, -4.04960880e+06])
    return {i: np.random.randint(low=i+1, high=len(demand)+1)
        for i, j in enumerate(demand)}

def gaussian_model(params, x):
    return params[2] * np.exp(-0.5*((x - params[0]) / params[1]) ** 2) + params[3]