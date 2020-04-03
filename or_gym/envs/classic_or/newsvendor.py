'''
Multi-period inventory management
Hector Perez, Christian Hubbs, Owais Sarwar
4/1/2020
'''

import gym
import itertools
import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import *

class NewsVendorEnv(gym.Env):
    '''
    The supply chain environment is structured as follows:
    
    It is a multiperiod multiechelon production-inventory system for a single non-perishable product that is sold only
    in discrete quantities. Each stage in the supply chain consists of an inventory holding area and a production area.
    The exception are the first stage (retailer: only inventory area) and the last stage (raw material transformation
    plant: only production area, with unlimited raw material availability). The inventory holding area holds the inventory
    necessary to produce the material at that stage. One unit of inventory produces one unit of product at each stage.
    There are lead times between the transfer of material from one stage to the next. The outgoing material from stage i 
    is the feed material for production at stage i-1. Stages are numbered in ascending order: Stages = {0, 1, ..., M} 
    (i.e. m = 0 is the retailer). Production at each stage is bounded by the stage's production capacity and the available
    inventory.
        
    At the each time period, the following sequence of events occurs:
    
    0) Stages 0 through M-1 place replinishment orders to their respective suppliers. Replenishment orders are filled
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
        dist_param = [dictionary] named values for parameters fed to statistical distribution.
            poisson: {'mu': <mean value>}
            binom: {'n': <mean value>, 'p': <probability between 0 and 1 of getting the mean value>}
            raindint: {'low' = <lower bound>, 'high': <upper bound>}
            geom: {'p': <probability. Outcome is the number of trials to success>}
        alpha = [float in range (0,1]] discount factor to account for the time value of money
        seed_int = [integer] seed for random state.
        '''
        #set default (arbitrary) values when creating environment (if no args or kwargs are given)
        self.periods = 1
        self.I0 = 1
        self.p = 1
        self.r = [0,0]
        self.k = [0,0]
        self.h = 0
        self.c = 1
        self.L = 0
        self.backlog = True
        self.dist = 1
        self.dist_param = {'mu': 1}
        self.alpha = 1
        self.seed_int = 0
        
        #add environment configuration dictionary and keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
        keys = ['periods','I0','p','r','k','h','c','L','backlog','dist','dist_param','alpha','seed_int']
        for i,value in enumerate(args):
            setattr(self,keys[i],value)
        if hasattr(self, 'env_config'):
            for key, value in self.env_config.items():
                setattr(self, key, value)
        
        #input parameters
        try:
            self.init_inv = np.array(list(self.I0))
        except:
            self.init_inv = np.array([self.I0])
        self.num_periods = self.periods
        self.unit_price = np.append(self.p,self.r[:-1]) #cost to stage 1 is price to stage 2
        self.unit_cost = np.array(self.r)
        self.demand_cost = np.array(self.k)
        self.holding_cost = np.append(self.h,0) #holding cost at last stage is 0
        try:
            self.supply_capacity = np.array(list(self.c))
        except:
            self.supply_capacity = np.array([self.c])
        try:
            self.lead_time = np.array(list(self.L))
        except:
            self.lead_time = np.array([self.L])
        self.backlog = self.backlog
        self.discount = self.alpha
        
        #intermediate calculation
        m = len(self.init_inv) + 1 #number of stages
        self.num_stages = m
        
        #parameters
        #dictionary with options for demand distributions
        distributions = {1:poisson,
                         2:binom,
                         3:randint,
                         4:geom}
        #distribution parameters
#         self.dist_param = self.dist_param
        
        #check inputs
        assert m >= 2, "The minimum number of stages is 2. Please try again"
        assert len(self.unit_cost) == m, "The length of r is not equal to the number of stages."
        assert len(self.demand_cost) == m, "The length of k is not equal to the number of stages."
        assert len(self.holding_cost) == m, "The length of h is not equal to the number of stages - 1."
        assert len(self.supply_capacity) == m-1, "The length of c is not equal to the number of stages - 1."
        assert len(self.lead_time) == m-1, "The length of L is not equal to the number of stages - 1."
        assert self.dist in [1,2,3,4], "dist must be one of 1, 2, 3, 4."
        assert distributions[self.dist].cdf(0,**self.dist_param), "Wrong parameters given for distribution."
        assert (self.alpha>0) & (self.alpha<=1), "alpha must be in the range (0,1]."
        
        #set random generation seed
        self.seed(self.seed_int) 
        
        #select distribution
        self.demand_dist = distributions[self.dist]  
        
        #intialize
        self.reset()
        
        #define spaces
        #action space (reorder quantities for each stage; list)
        self.action_space = gym.spaces.Box(low=0, high=np.max(self.supply_capacity), shape = (m-1,)) #an action is defined for every stage (except last one)
        #observation space (Inventory position at each echelon, which is any integer value)
        self.observation_space = gym.spaces.Box(low=-np.Inf, high=np.Inf, shape = (m-1,))#, dtype=np.int32)
        
    def seed(self,seed=None):
        '''
        Set random number generation seed
        '''
        #seed random state
        if seed != None:
            np.random.seed(seed=int(seed))
        
    def reset(self):
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
        
        #simulation result lists
        self.I=np.zeros([periods + 1, m - 1]) #inventory at the beginning of each period (last stage not included since iventory is infinite)
        self.T=np.zeros([periods + 1, m - 1]) #pipeline inventory at the beginning of each period (no pipeline inventory for last stage)
        self.R=np.zeros([periods, m - 1]) #replenishment order (last stage places no replenishment orders)
        self.D=np.zeros(periods) #demand at retailer
        self.S=np.zeros([periods, m]) #units sold
        self.B=np.zeros([periods, m]) #backlog (includes top most production site in supply chain)
        self.LS=np.zeros([periods, m]) #lost sales
        self.P=np.zeros(periods) #profit
        
        #initializetion
        self.period = 0 #initialize time
        self.I[0,:]=np.array(I0) #initial inventory
        self.T[0,:]=np.zeros(m-1) #initial pipeline inventory
        
        #set state
        self._update_state()
        
        return self.state
    
    def _update_state(self):
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
    
    def step(self,action):
        '''
        Take a step in time in the multiperiod inventory management problem.
        action = [integer; dimension |Stages|-1] number of units to request from suppliers (last stage makes no requests)
        '''
        #get inventory at hand and pipeline inventory at beginning of the period
        n = self.period
        L = self.lead_time
        I = self.I[n,:].copy() #inventory at start of period n
        T = self.T[n,:].copy() #pipeline inventory at start of period n
        m = self.num_stages #number of stages
        
        #get production capacities
        c = self.supply_capacity #capacity
        
        #available inventory at the m+1 stage (note: last stage has unlimited supply)
        Im1 = np.append(I[1:], np.Inf) 
        
        #place replenishment order
        R = action.astype(int)
        R[R<0] = 0 #force non-negativity
        if n>=1: #add backlogged replenishment orders to current request
            R = R + self.B[n-1,1:]
        Rcopy = R #copy oritignal replenishment quantity
        R[R>=c] = c[R>=c] #enforce capacity constraint
        R[R>=Im1] = Im1[R>=Im1] #enforce available inventory constraint
        self.R[n,:] = R #store R[n]
        
        #receive inventory replenishment placed L periods ago
        RnL = np.zeros(m-1) #initialize
        for i in range(m-1):
            if n - L[i] >= 0:
                RnL[i] = self.R[n-L[i],i].copy() #replenishment placed at the end of period n-L-1
                I[i] = I[i] + RnL[i]
            
        #demand is realized
        D0 = self.demand_dist.rvs(**self.dist_param)
        D = D0 #demand
        self.D[n] = D0 #store D[n]
        
        #add previous backlog to demand
        if n >= 1:
            D = D0 + self.B[n-1,0].copy() #add backlogs to demand
        
        #units sold
        S0 = min(I[0],D) #at retailer
        S = np.append(S0,R) #at each stage
        self.S[n,:] = S #store S[n]
        
        #update inventory at hand and pipeline inventory
        I = I - S[:-1] #updated inventory at all stages (exclude last stage)
        T = T - RnL + R #updated pipeline inventory at all stages (exclude last one)
        self.I[n+1,:] = I #store inventory available at start of period n + 1 (exclude last stage)
        self.T[n+1,:] = T #store pipeline inventory at start of period n + 1
        
        #unfulfilled orders
        U = np.append(D,Rcopy) - S #unfulfilled demand and replenishment orders
        
        #backlog and lost sales
        if self.backlog:
            B = U
            LS = np.zeros(m) 
        else:
            LS = U #lost sales
            B = np.zeros(m)
        self.B[n,:] = B #store B[n]
        self.LS[n,:] = LS #store LS[n]

        #calculate profit
        p = self.unit_price 
        r = self.unit_cost 
        k = self.demand_cost
        h = self.holding_cost
        a = self.discount
        II = np.append(I,0) #augment inventory so that last has no onsite inventory
        RR = np.append(R,S[-1]) #augment replenishment orders to include production cost at last stage
        P = a**n*np.sum(p*S - (r*RR + k*U + h*II)) #discounted profit in period n
        self.P[n] = P #store P
        
        #update period
        self.period += 1  
        
        #update stae
        self._update_state()
        
        #return values
        reward = P #profit at current period
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
        IP = self.state #extract inventory position (current state)
        
        try:
            dimz = len(z)
        except:
            dimz = 1
        assert dimz == m-1, "Wrong dimension on base stock level vector. Should be #Stages - 1."
        
        #calculate total inventory position at the beginning of period n
        R = z - IP #replenishmet order to reach zopt

        #check if R can actually be fulfilled (capacity and inventory constraints)
        Im1 = np.append(self.I[n,1:], np.Inf) #available inventory at the m+1 stage
                                            #NOTE: last stage has unlimited raw materials
        Rpos = np.column_stack((np.zeros(len(R)),R)) #augmented materix to get replenishment only if positive
        A = np.column_stack((c, np.max(Rpos,axis=1), Im1)) #augmented matrix with c, R, and I_m+1 as columns
        
        R = np.min(A, axis = 1) #replenishmet order to reach zopt (capacity constrained)
        
        return R