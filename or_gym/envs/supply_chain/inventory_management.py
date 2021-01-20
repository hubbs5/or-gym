'''
Multi-period inventory management
Hector Perez, Christian Hubbs, Owais Sarwar
4/14/2020
'''

import gym
import itertools
import numpy as np
from scipy.stats import *
from or_gym.utils import assign_env_config
from collections import deque

class InvManagementMasterEnv(gym.Env):
    '''
    The supply chain environment is structured as follows:
    
    It is a multi-period multi-echelon production-inventory system for a single non-perishable product that is sold only
    in discrete quantities. Each stage in the supply chain consists of an inventory holding area and a production area.
    The exception are the first stage (retailer: only inventory area) and the last stage (raw material transformation
    plant: only production area, with unlimited raw material availability). The inventory holding area holds the inventory
    necessary to produce the material at that stage. One unit of inventory produces one unit of product at each stage.
    There are lead times between the transfer of material from one stage to the next. The outgoing material from stage i 
    is the feed material for production at stage i-1. Stages are numbered in ascending order: Stages = {0, 1, ..., M} 
    (i.e. m = 0 is the retailer). Production at each stage is bounded by the stage's production capacity and the available
    inventory.
        
    At the beginning of each time period, the following sequence of events occurs:
    
    0) Stages 0 through M-1 place replenishment orders to their respective suppliers. Replenishment orders are filled
        according to available production capacity and available inventory at the respective suppliers.
    1) Stages 0 through M-1 receive incoming inventory replenishment shipments that have made it down the product pipeline
        after the stage's respective lead time.
    2) Customer demand occurs at stage 0 (retailer). It is sampled from a specified discrete probability distribution.
    3) Demand is filled according to available inventory at stage 0.
    4) Option: one of the following occurs,
        a) Unfulfilled sales and replenishment orders are backlogged at a penalty. 
            Note: Backlogged sales take priority in the following period.
        b) Unfulfilled sales and replenishment orders are lost with a goodwill loss penalty. 
    5) Surpluss inventory is held at each stage at a holding cost.
        
    '''
    def __init__(self, *args, **kwargs):
        '''
        periods = [positive integer] number of periods in simulation.
        I0 = [non-negative integer; dimension |Stages|-1] initial inventories for each stage.
        p = [positive float] unit price for final product.
        r = [non-negative float; dimension |Stages|] unit cost for replenishment orders at each stage.
        k = [non-negative float; dimension |Stages|] backlog cost or goodwill loss (per unit) for unfulfilled orders (demand or replenishment orders).
        h = [non-negative float; dimension |Stages|-1] unit holding cost for excess on-hand inventory at each stage.
            (Note: does not include pipeline inventory).
        c = [positive integer; dimension |Stages|-1] production capacities for each suppliers (stages 1 through |Stage|).
        L = [non-negative integer; dimension |Stages|-1] lead times in betwen stages.
        backlog = [boolean] are unfulfilled orders backlogged? True = backlogged, False = lost sales.
        dist = [integer] value between 1 and 4. Specifies distribution for customer demand.
            1: poisson distribution
            2: binomial distribution
            3: uniform random integer
            4: geometric distribution
            5: user supplied demand values
        dist_param = [dictionary] named values for parameters fed to statistical distribution.
            poisson: {'mu': <mean value>}
            binom: {'n': <mean value>, 'p': <probability between 0 and 1 of getting the mean value>}
            raindint: {'low' = <lower bound>, 'high': <upper bound>}
            geom: {'p': <probability. Outcome is the number of trials to success>}
        alpha = [float in range (0,1]] discount factor to account for the time value of money
        seed_int = [integer] seed for random state.
        user_D = [list] user specified demand for each time period in simulation
        '''
        # set default (arbitrary) values when creating environment (if no args or kwargs are given)
        self.periods = 30
        self.I0 = [100, 100, 200]
        self.p = 2
        self.r = [1.5, 1.0, 0.75, 0.5]
        self.k = [0.10, 0.075, 0.05, 0.025]
        self.h = [0.15, 0.10, 0.05]
        self.c = [100, 90, 80]
        self.L = [3, 5, 10]
        self.backlog = True
        self.dist = 1
        self.dist_param = {'mu': 20}
        self.alpha = 0.97
        self.seed_int = 0
        self.user_D = np.zeros(self.periods)
        self._max_rewards = 2000
        
        # add environment configuration dictionary and keyword arguments
        assign_env_config(self, kwargs)
        
        # input parameters
        try:
            self.init_inv = np.array(list(self.I0))
        except:
            self.init_inv = np.array([self.I0])
        self.num_periods = self.periods
        self.unit_price = np.append(self.p,self.r[:-1]) # cost to stage 1 is price to stage 2
        self.unit_cost = np.array(self.r)
        self.demand_cost = np.array(self.k)
        self.holding_cost = np.append(self.h,0) # holding cost at last stage is 0
        try:
            self.supply_capacity = np.array(list(self.c))
        except:
            self.supply_capacity = np.array([self.c])
        try:
            self.lead_time = np.array(list(self.L))
        except:
            self.lead_time = np.array([self.L])
        self.discount = self.alpha
        self.user_D = np.array(list(self.user_D))
        self.num_stages = len(self.init_inv) + 1
        m = self.num_stages
        lt_max = self.lead_time.max()
        
        #  parameters
        #  dictionary with options for demand distributions
        distributions = {1:poisson,
                         2:binom,
                         3:randint,
                         4:geom,
                         5:self.user_D}

        # check inputs
        assert np.all(self.init_inv) >=0, "The initial inventory cannot be negative"
        try:
            assert self.num_periods > 0, "The number of periods must be positive. Num Periods = {}".format(self.num_periods)
        except TypeError:
            print('\n{}\n'.format(self.num_periods))
        assert np.all(self.unit_price >= 0), "The sales prices cannot be negative."
        assert np.all(self.unit_cost >= 0), "The procurement costs cannot be negative."
        assert np.all(self.demand_cost >= 0), "The unfulfilled demand costs cannot be negative."
        assert np.all(self.holding_cost >= 0), "The inventory holding costs cannot be negative."
        assert np.all(self.supply_capacity > 0), "The supply capacities must be positive."
        assert np.all(self.lead_time >= 0), "The lead times cannot be negative."
        assert (self.backlog == False) | (self.backlog == True), "The backlog parameter must be a boolean."
        assert m >= 2, "The minimum number of stages is 2. Please try again"
        assert len(self.unit_cost) == m, "The length of r is not equal to the number of stages."
        assert len(self.demand_cost) == m, "The length of k is not equal to the number of stages."
        assert len(self.holding_cost) == m, "The length of h is not equal to the number of stages - 1."
        assert len(self.supply_capacity) == m-1, "The length of c is not equal to the number of stages - 1."
        assert len(self.lead_time) == m-1, "The length of L is not equal to the number of stages - 1."
        assert self.dist in [1,2,3,4,5], "dist must be one of 1, 2, 3, 4, 5."
        if self.dist < 5:
            assert distributions[self.dist].cdf(0,**self.dist_param), "Wrong parameters given for distribution."
        else:
            assert len(self.user_D) == self.num_periods, "The length of the user specified distribution is not equal to the number of periods."
        assert (self.alpha>0) & (self.alpha<=1), "alpha must be in the range (0,1]."
        
        # select distribution
        self.demand_dist = distributions[self.dist]  
        
        # set random generation seed (unless using user demands)
        if self.dist < 5:
            self.seed(self.seed_int)

        # intialize
        self.reset()
        
        # action space (reorder quantities for each stage; list)
        # An action is defined for every stage (except last one)
        # self.action_space = gym.spaces.Tuple(tuple(
            # [gym.spaces.Box(0, i, shape=(1,)) for i in self.supply_capacity]))
        self.pipeline_length = (m-1)*(lt_max+1)
        self.action_space = gym.spaces.Box(
            low=np.zeros(m-1), high=self.supply_capacity, dtype=np.int16)
        # observation space (Inventory position at each echelon, which is any integer value)
        self.observation_space = gym.spaces.Box(
            low=-np.ones(self.pipeline_length)*self.supply_capacity.max()*self.num_periods*10,
            high=np.ones(self.pipeline_length)*self.supply_capacity.max()*self.num_periods, dtype=np.int32)

        # self.observation_space = gym.spaces.Box(
        #     low=-np.ones(m-1)*self.supply_capacity.max()*self.num_periods*10, 
        #     high=self.supply_capacity*self.num_periods, dtype=np.int32)

    def seed(self,seed=None):
        '''
        Set random number generation seed
        '''
        # seed random state
        if seed != None:
            np.random.seed(seed=int(seed))
        
    def _RESET(self):
        '''
        Create and initialize all variables and containers.
        Nomenclature:
            I = On hand inventory at the start of each period at each stage (except last one).
            T = Pipeline inventory at the start of each period at each stage (except last one).
            R = Replenishment order placed at each period at each stage (except last one).
            D = Customer demand at each period (at the retailer)
            S = Sales performed at each period at each stage.
            B = Backlog at each period at each stage.
            LS = Lost sales at each period at each stage.
            P = Total profit at each stage.
        '''
        periods = self.num_periods
        m = self.num_stages
        I0 = self.init_inv
        
        # simulation result lists
        self.I=np.zeros([periods + 1, m - 1]) # inventory at the beginning of each period (last stage not included since iventory is infinite)
        self.T=np.zeros([periods + 1, m - 1]) # pipeline inventory at the beginning of each period (no pipeline inventory for last stage)
        self.R=np.zeros([periods, m - 1]) # replenishment order (last stage places no replenishment orders)
        self.D=np.zeros(periods) # demand at retailer
        self.S=np.zeros([periods, m]) # units sold
        self.B=np.zeros([periods, m]) # backlog (includes top most production site in supply chain)
        self.LS=np.zeros([periods, m]) # lost sales
        self.P=np.zeros(periods) # profit
        
        # initializetion
        self.period = 0 # initialize time
        self.I[0,:]=np.array(I0) # initial inventory
        self.T[0,:]=np.zeros(m-1) # initial pipeline inventory 
        self.action_log = np.zeros((periods, m-1))
        # set state
        self._update_state()
        
        return self.state

    def _update_state(self):
        m = self.num_stages - 1
        t = self.period
        lt_max = self.lead_time.max()
        state = np.zeros(m*(lt_max + 1))
        # state = np.zeros(m)
        if t == 0:
            state[:m] = self.I0
        else:
            state[:m] = self.I[t]

        if t == 0:
            pass
        elif t >= lt_max:
            state[-m*lt_max:] += self.action_log[t-lt_max:t].flatten()
        else:
            state[-m*(t):] += self.action_log[:t].flatten()

        self.state = state.copy()
    
    def _update_base_stock_policy_state(self):
        '''
        Get current state of the system: Inventory position at each echelon
        Inventory at hand + Pipeline inventory - backlog up to the current stage 
        (excludes last stage since no inventory there, nor replenishment orders placed there).
        '''
        n = self.period
        m = self.num_stages
        if n>=1:
            IP = np.cumsum(self.I[n,:] + self.T[n,:] - self.B[n-1,:-1])
        else:
            IP = np.cumsum(self.I[n,:] + self.T[n,:])
        self.state = IP
    
    def _STEP(self,action):
        '''
        Take a step in time in the multiperiod inventory management problem.
        action = [integer; dimension |Stages|-1] number of units to request from suppliers (last stage makes no requests)
        '''
        R = np.maximum(action, 0).astype(int)

        # get inventory at hand and pipeline inventory at beginning of the period
        n = self.period
        L = self.lead_time
        I = self.I[n,:].copy() # inventory at start of period n
        T = self.T[n,:].copy() # pipeline inventory at start of period n
        m = self.num_stages # number of stages
        
        # get production capacities
        c = self.supply_capacity # capacity
        self.action_log[n] = R.copy()
        # available inventory at the m+1 stage (note: last stage has unlimited supply)
        Im1 = np.append(I[1:], np.Inf) 
        
        # place replenishment order
        if n>=1: # add backlogged replenishment orders to current request
            R = R + self.B[n-1,1:]
        Rcopy = R.copy() # copy original replenishment quantity
        R[R>=c] = c[R>=c] # enforce capacity constraint
        R[R>=Im1] = Im1[R>=Im1] # enforce available inventory constraint
        self.R[n,:] = R # store R[n]
        
        # receive inventory replenishment placed L periods ago
        RnL = np.zeros(m-1) # initialize
        for i in range(m-1):
            if n - L[i] >= 0:
                RnL[i] = self.R[n-L[i],i].copy() # replenishment placed at the end of period n-L-1
                I[i] = I[i] + RnL[i]
            
        # demand is realized
        if self.dist < 5:
            D0 = self.demand_dist.rvs(**self.dist_param)
        else:
            D0 = self.demand_dist[n] # user specified demand
        D = D0 # demand
        self.D[n] = D0 # store D[n]
        
        # add previous backlog to demand
        if n >= 1:
            D = D0 + self.B[n-1,0].copy() # add backlogs to demand
        
        # units sold
        S0 = min(I[0],D) # at retailer
        S = np.append(S0,R) # at each stage
        self.S[n,:] = S # store S[n]
        
        # update inventory on hand and pipeline inventory
        I = I - S[:-1] # updated inventory at all stages (exclude last stage)
        T = T - RnL + R # updated pipeline inventory at all stages (exclude last one)
        self.I[n+1,:] = I # store inventory available at start of period n + 1 (exclude last stage)
        self.T[n+1,:] = T # store pipeline inventory at start of period n + 1
        
        # unfulfilled orders
        U = np.append(D, Rcopy) - S # unfulfilled demand and replenishment orders
        
        # backlog and lost sales
        if self.backlog:
            B = U
            LS = np.zeros(m)
        else:
            LS = U # lost sales
            B = np.zeros(m)
        self.B[n,:] = B # store B[n]
        self.LS[n,:] = LS # store LS[n]

        # calculate profit
        p = self.unit_price 
        r = self.unit_cost 
        k = self.demand_cost
        h = self.holding_cost
        a = self.discount
        II = np.append(I,0) # augment inventory so that last has no onsite inventory
        RR = np.append(R,S[-1]) # augment replenishment orders to include production cost at last stage
        P = a**n*np.sum(p*S - (r*RR + k*U + h*II)) # discounted profit in period n
        # P = a**n*np.sum(p*S - (r*RR + k*U + h*I))
        self.P[n] = P # store P
        
        # update period
        self.period += 1  
        
        # update stae
        self._update_state()
        
        # set reward (profit from current timestep)
        reward = P 
        
        # determine if simulation should terminate
        if self.period >= self.num_periods:
            done = True
        else:
            done = False
            
        return self.state, reward, done, {}
    
    def sample_action(self):
        '''
        Generate an action by sampling from the action_space
        '''
        return self.action_space.sample()
        
    def base_stock_action(self,z):
        '''
        Sample action (number of units to request) based on a base-stock policy (order up to z policy)
        z = [integer list; dimension |Stages| - 1] base stock level (no inventory at the last stage)
        '''
        n = self.period
        c = self.supply_capacity
        m = self.num_stages
        IP = self._update_base_stock_policy_state() # extract inventory position (current state)
        
        try:
            dimz = len(z)
        except:
            dimz = 1
        assert dimz == m-1, "Wrong dimension on base stock level vector. Should be # Stages - 1."
        
        # calculate total inventory position at the beginning of period n
        R = z - IP # replenishmet order to reach zopt

        # check if R can actually be fulfilled (capacity and inventory constraints)
        Im1 = np.append(self.I[n,1:], np.Inf) # available inventory at the m+1 stage
                                            # NOTE: last stage has unlimited raw materials
        Rpos = np.column_stack((np.zeros(len(R)),R)) # augmented materix to get replenishment only if positive
        A = np.column_stack((c, np.max(Rpos,axis=1), Im1)) # augmented matrix with c, R, and I_m+1 as columns
        
        R = np.min(A, axis = 1) # replenishmet order to reach zopt (capacity constrained)
        
        return R

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()
        
class InvManagementBacklogEnv(InvManagementMasterEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class InvManagementLostSalesEnv(InvManagementMasterEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backlog = False
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.pipeline_length), # Never goes negative without backlog
            high=np.ones(self.pipeline_length)*self.supply_capacity.max()*self.num_periods, dtype=np.int32)