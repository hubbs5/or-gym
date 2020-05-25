#!usr/bin/env python

import gym
import or_gym
import pyomo.environ as pe
import numpy as np

def build_im_mip_model(env,bigm=10000,online=False):
    '''
    Optimize base stock level (z variable) on a simulated sample path using an MILP. The existing
    sample path (historical demands) is used when running online.
    
    Notes: 
        -z is constant in time (static base-stock policy)
        -Using the hull reformulation instead of big-M could speed things up. Using tighter 
            big-M could also be helpful.
        -All parameters to the simulation environment must have been defined 
            previously when making the environment.
    
    env = [InvManagementEnv] current simulation environment. 
    bigm = [Float] big-M value for BM reformulation
    online = [Boolean] should the optimization be run online?
    ''' 
    
    # assert env.spec.id == 'InvManagement-v0', \
        # '{} received. Heuristic designed for InvManagement-v0.'.format(env.spec.id)
    #do not reset environment
    
    #big m values
    M = bigm
    BigM1 = M
    BigM2 = M
    BigM3 = -M
    BigM4 = -M
    BigM5 = -M
    BigM6 = -M
    
    #create model
    mip = pe.ConcreteModel()
    
    #define sets
    if online:
        mip.n = pe.RangeSet(0,env.period-1) #periods
        mip.n1 = pe.RangeSet(0,env.period) #periods (includes an additional period for final inventories)
    else:
        mip.n = pe.RangeSet(0,env.num_periods-1) 
        mip.n1 = pe.RangeSet(0,env.num_periods)
    mip.m = pe.RangeSet(0,env.num_stages-1) #stages
    mip.m0 = pe.RangeSet(0,env.num_stages-2) #stages (excludes last stage which has no inventory)
    
    #define parameters
    mip.unit_price = pe.Param(mip.m, initialize = {i:env.unit_price[i] for i in mip.m}) #sales price for each stage
    mip.unit_cost = pe.Param(mip.m, initialize = {i:env.unit_cost[i] for i in mip.m}) #purchasing cost for each stage
    mip.demand_cost = pe.Param(mip.m, initialize = {i:env.demand_cost[i] for i in mip.m}) #cost for unfulfilled demand at each stage
    mip.holding_cost = pe.Param(mip.m, initialize = {i:env.holding_cost[i] for i in mip.m}) #inventory holding cost at each stage
    mip.supply_capacity = pe.Param(mip.m0, initialize = {i:env.supply_capacity[i] for i in mip.m0}) #production capacity at each stage
    mip.lead_time = pe.Param(mip.m0, initialize = {i:env.lead_time[i] for i in mip.m0}) #lead times in between stages
    mip.discount = env.discount #time-valued discount 
    backlog = env.backlog #backlog or lost sales
    if online: #only use up to the current period
        mip.num_periods = env.period
        D = env.D[:env.period]
    else: #use full simulation if offline
        mip.num_periods = env.num_periods
        D = env.demand_dist.rvs(size=env.num_periods,**env.dist_param)
    mip.D = pe.Param(mip.n, initialize = {i:D[i] for i in mip.n}) #store demands
    prob = env.demand_dist.pmf(D,**env.dist_param) #probability of each demand based on distribution
    mip.prob = pe.Param(mip.n, initialize = {i:prob[i] for i in mip.n}) #store probability at each period
    
    #define variables
    mip.I = pe.Var(mip.n1,mip.m0,domain=pe.NonNegativeReals) #on hand inventory at each stage
    mip.T = pe.Var(mip.n1,mip.m0,domain=pe.NonNegativeReals) #pipeline inventory in between each stage
    mip.R = pe.Var(mip.n,mip.m0,domain=pe.NonNegativeReals) #reorder quantities for each stage
    mip.R1 = pe.Var(mip.n,mip.m0,domain=pe.NonNegativeReals) #unconstrained reorder quantity
    mip.S = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals) #sales at each stage
    if backlog:
        mip.B = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals) #backlogs at each stage
    else:
        mip.LS = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals) #lost sales at each stage
    mip.P = pe.Var(mip.n,domain=pe.Reals) #profit at each stage
    mip.y = pe.Var(mip.n,mip.m0,domain=pe.Binary) #auxiliary variable (y = 0: inventory level is above the base stock level (no reorder))
    mip.y1 = pe.Var(mip.n,mip.m0,domain=pe.Binary) #auxiliary variable (y1 = 1: unconstrained reorder quantity accepted)
    mip.y2 = pe.Var(mip.n,mip.m0,domain=pe.Binary) #auxiliary variable (y2 = 1: reorder quantity is capacity constrained)
    if env.num_stages > 2:
        mip.y3 = pe.Var(mip.n,mip.m0,domain=pe.Binary) #auxiliary variable (y3 = 1: reorder quantity is inventory constrained)
    mip.y4 = pe.Var(mip.n,mip.m,domain=pe.Binary) #auxiliary variable (y4 = 1: demand + backlog satisfied)
    mip.x = pe.Var(mip.m0,domain=pe.NonNegativeReals) #inventory level at each stage
    mip.z = pe.Var(mip.m0,domain=pe.PositiveIntegers) #base stock level at each stage
    
    #initialize
    for m in mip.m0:
        mip.T[0,m].fix(0)
    
    #define constraints
    mip.inv_bal = pe.ConstraintList()
    mip.sales1 = pe.ConstraintList()
    mip.sales2 = pe.ConstraintList()
    mip.sales3 = pe.ConstraintList()
    mip.sales4 = pe.ConstraintList()
    mip.sales5 = pe.ConstraintList()
    mip.reorder1 = pe.ConstraintList()
    mip.reorder2 = pe.ConstraintList()
    mip.reorder3 = pe.ConstraintList()
    mip.reorder4 = pe.ConstraintList()
    mip.reorder5 = pe.ConstraintList()
    mip.reorder6= pe.ConstraintList()
    mip.reorder7= pe.ConstraintList()
    mip.reorder8= pe.ConstraintList()
    mip.reorder9= pe.ConstraintList()
    mip.reorder10 =  pe.ConstraintList()
    mip.pip_bal = pe.ConstraintList()
    mip.unfulfilled = pe.ConstraintList()
    mip.profit = pe.ConstraintList()
    mip.basestock = pe.ConstraintList()
    mip.init_inv= pe.ConstraintList()
    
    #build constraints
    for m in mip.m0:
        #relate base stock levels to inventory levels: base stock level = total inventory up to that echelon
        mip.basestock.add(mip.z[m] == sum(mip.x[i] for i in range(m+1)))
        #initialize inventory levels to being full (this would be ideal)
        mip.init_inv.add(mip.I[0,m] == mip.x[m])
    
    for n in mip.n:
        #calculate profit: apply time value discount to sales revenue - purchasing costs - unfulfilled demand cost - holding cost
        if backlog:
            mip.profit.add(mip.P[n] == mip.discount**n * (sum(mip.unit_price[m]*mip.S[n,m] for m in mip.m)
                                                    - (sum(mip.unit_cost[m]*mip.R[n,m] for m in mip.m0) + mip.unit_cost[mip.m[-1]]*mip.S[n,mip.m[-1]])
                                                    - sum(mip.demand_cost[m]*mip.B[n,m] for m in mip.m)
                                                    - sum(mip.holding_cost[m]*mip.I[n+1,m] for m in mip.m0)))
        else:
            mip.profit.add(mip.P[n] == mip.discount**n * (sum(mip.unit_price[m]*mip.S[n,m] for m in mip.m)
                                                    - (sum(mip.unit_cost[m]*mip.R[n,m] for m in mip.m0) + mip.unit_cost[mip.m[-1]]*mip.S[n,mip.m[-1]])
                                                    - sum(mip.demand_cost[m]*mip.LS[n,m] for m in mip.m)
                                                    - sum(mip.holding_cost[m]*mip.I[n+1,m] for m in mip.m0)))
            
        for m in mip.m0:
            #on-hand inventory balance: next period inventory = prev period inventory + arrival from above stage - sales
            if n - mip.lead_time[m] >= 0:
                mip.inv_bal.add(mip.I[n+1,m] == mip.I[n,m] + mip.R[n - mip.lead_time[m],m] - mip.S[n,m])
            else:
                mip.inv_bal.add(mip.I[n+1,m] == mip.I[n,m] - mip.S[n,m])
            #pipeline inventory balance: next period inventory = prev period inventory - delivered material + new reorder
            if n - mip.lead_time[m] >= 0:
                mip.pip_bal.add(mip.T[n+1,m] == mip.T[n,m] - mip.R[n - mip.lead_time[m],m] + mip.R[n,m])
            else:
                mip.pip_bal.add(mip.T[n+1,m] == mip.T[n,m] + mip.R[n,m])
            #reorder quantity constraints: R1 = max(0, z - sum(I + T - B) + B[m+1])
                # reorder based on base_stock level = z - sum(I + T - B)
                # Note: R1 = max(A,B) <-> A <= R1 <= A + M*(1-y) ;  B <= R1 <= B + M*y
                # y = 1 means that the reorder level is positive
                # y = 0 means that no reorder necessary (already above base stock level)
                # Disjunction: [y -> R1 = z - sum(I + T - B)] OR [not y -> R1 = 0]
            if (backlog) & (n-1>=0):
                mip.reorder1.add(mip.R1[n,m] <= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] - mip.B[n-1,i] for i in range(m+1)) + mip.B[n-1,m+1] + BigM1 * (1 - mip.y[n,m]))
                mip.reorder2.add(mip.R1[n,m] >= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] - mip.B[n-1,i] for i in range(m+1)) + mip.B[n-1,m+1])
            else:
                mip.reorder1.add(mip.R1[n,m] <= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] for i in range(m+1)) + BigM1 * (1 - mip.y[n,m]))
                mip.reorder2.add(mip.R1[n,m] >= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] for i in range(m+1)))
            mip.reorder3.add(mip.R1[n,m] <= BigM2 * mip.y[n,m])
            #reorder quantity constraints: R = min(c, I[m+1], R1)
                #last constraint ensures that only one of the 3 options is chosen
                # Note: R = min(A,B,C) <-> A + M*(1-y1) <= R <= A  ;  B + M*(1-y2) <= R <= B  ;  C + M*(1-y3) <= R <= C  ;  y1+y2+y3==1
            mip.reorder4.add(mip.R[n,m] <= mip.R1[n,m])
            mip.reorder5.add(mip.R[n,m] >= mip.R1[n,m] + BigM3 * (1 - mip.y1[n,m]))
            mip.reorder6.add(mip.R[n,m] <= mip.supply_capacity[m])
            mip.reorder7.add(mip.R[n,m] >= mip.supply_capacity[m] * mip.y2[n,m])
            if (m < mip.m0[-1]) & (env.num_stages > 2): 
                #if number of stages = 2, then there is no inventory constraint since the last level has unlimited inventory
                #also, last stage has unlimited inventory
                mip.reorder8.add(mip.R[n,m] <= mip.I[n,m+1])
                mip.reorder9.add(mip.R[n,m] >= mip.I[n,m+1] + BigM4 * (1 - mip.y3[n,m]))
                mip.reorder10.add(mip.y1[n,m] + mip.y2[n,m] + mip.y3[n,m] == 1)
            else:
                mip.reorder10.add(mip.y1[n,m] + mip.y2[n,m] == 1)
                
        for m in mip.m:            
            if m == 0:
            #sales constraints: S = min(I + R[n-L], D + B[n-1]) at stage 0
                if n - mip.lead_time[m] >= 0:
                    mip.sales1.add(mip.S[n,m] <= mip.I[n,m] + mip.R[n - mip.lead_time[m],m])
                    mip.sales2.add(mip.S[n,m] >= mip.I[n,m] + mip.R[n - mip.lead_time[m],m] + BigM5 * (1 - mip.y4[n,m]))
                else:
                    mip.sales1.add(mip.S[n,m] <= mip.I[n,m])
                    mip.sales2.add(mip.S[n,m] >= mip.I[n,m] + BigM5 * (1 - mip.y4[n,m]))
                
                if (backlog) & (n-1>=0):
                    mip.sales3.add(mip.S[n,m] <= mip.D[n] + mip.B[n-1,m])
                    mip.sales4.add(mip.S[n,m] >= mip.D[n] + mip.B[n-1,m] + BigM6 * mip.y4[n,m])
                else:
                    mip.sales3.add(mip.S[n,m] <= mip.D[n])
                    mip.sales4.add(mip.S[n,m] >= mip.D[n] + BigM6 * mip.y4[n,m])
            else:
            #sales constraints: S = R[n,m-1] at higher level stages
                mip.sales5.add(mip.S[n,m] == mip.R[n,m-1])
                    
            if m == 0:
            #unfulfilled orders at stage 0: U = D + B[n-1] - S
                if backlog:
                    if n-1>=0:
                        mip.unfulfilled.add(mip.B[n,m] == mip.D[n] + mip.B[n-1,m] - mip.S[n,m])
                    else:
                        mip.unfulfilled.add(mip.B[n,m] == mip.D[n] - mip.S[n,m])
                else:
                    mip.unfulfilled.add(mip.LS[n,m] == mip.D[n] - mip.S[n,m])
            else:
            #unfulfilled orders at stage higher level stages: U = R[n,m-1] + B[n-1,m] - S[n,m]
                if backlog:
                    if n-1>=0:
                        mip.unfulfilled.add(mip.B[n,m] == mip.R[n,m-1] + mip.B[n-1,m] - mip.S[n,m])
                    else:
                        mip.unfulfilled.add(mip.B[n,m] == mip.R[n,m-1] - mip.S[n,m])
                else:
                    mip.unfulfilled.add(mip.LS[n,m] == mip.R[n,m-1] - mip.S[n,m])

    #objective function: maximize expected profit
    mip.obj = pe.Objective(
        expr = 1/mip.num_periods * sum(mip.prob[n]*mip.P[n] for n in mip.n),
        sense = pe.maximize)
    
    return mip

def im_dfo_model(x,env,online):
    '''
    Compute negative of the expected profit for a sample path.
    This function is used in an unconstrained optimization algorithm (scipy.optimize.minimize).
    
    x = [integer list; dimension |Stages| - 1] total inventory levels at each node.
    env = [InvManagementEnv] current simulation environment.
    online = [Boolean] should the optimization be run online?
    '''
    
    # assert env.spec.id == 'InvManagement-v0', \
        # '{} received. Heuristic designed for InvManagement-v0.'.format(env.spec.id)
    
    x = np.array(x) #inventory level at each node
    z = np.cumsum(x) #base stock levels
    
    m = env.num_stages
    try:
        dimz = len(z)
    except:
        dimz = 1
    assert dimz == m-1, "Wrong dimension on base stock level vector. Should be #Stages - 1."
    
    #create simulation environment (copy it if in offline mode)
    sim_kwargs = {'I0': x, #set initial inventory to full base stock
                  'p': env.p, #extract all other parameters from env
                  'r': env.r,
                  'k': env.k,
                  'h': env.h,
                  'c': env.c,
                  'L': env.L,
                  'backlog': env.backlog,
                  'dist_param': env.dist_param,
                  'alpha': env.alpha,
                  'seed_int': env.seed_int}
                  
    demand_dist = env.demand_dist #extract demand distribution function from env
    
    if online:
        #extract args to pass to re-simulation
        sim_kwargs['periods'] = env.period #simulation goes up until current period in online mode
        sim_kwargs['dist'] = 5 #set distribution to manual mode
        sim_kwargs['user_D'] = env.D[:env.period] #copy historical demands from env  
    else:
        sim_kwargs['periods'] = env.num_periods #copy num_periods from env
        sim_kwargs['dist'] = env.dist #copy dist from env
        
    #build simulation environment (this is just clean copy if in offline mode)
    if env.backlog:
        sim = or_gym.make("InvManagement-v0",env_config=sim_kwargs)
    else:
        sim = or_gym.make("InvManagement-v1",env_config=sim_kwargs)
    
    #run simulation
    for t in range(sim.num_periods):
        #take a step in the simulation using critical ratio base stock
        sim.step(action=sim.base_stock_action(z=z)) 
    
    #probability for demand at each period
    prob = demand_dist.pmf(sim.D,**sim.dist_param) 
    
    #expected profit
    return -1/sim.num_periods*np.sum(prob*sim.P)
    
def build_im_pi_mip_model(env):#,bigm=10000):
    '''
    Build a perfect information MILP model for the InvManagement problem. No policy is used for the reorder.
    This will give you the optimal reorder quantities if you knew the demand before hand.
    
    Notes: 
        # -Using the hull reformulation instead of big-M could speed things up. Using tighter 
            # big-M could also be helpful.
        -All parameters to the simulation environment must have been defined 
            previously when making the environment.
    
    env = [InvManagementEnv] current simulation environment. 
    # bigm = [Float] big-M value for BM reformulation
    ''' 
    
    # assert env.spec.id == 'InvManagement-v0', \
        # '{} received. Heuristic designed for InvManagement-v0.'.format(env.spec.id)
    #do not reset environment
    
    #big m values
    # M = bigm
    # BigM1 = M
    # BigM2 = M
    # BigM3 = -M
    # BigM4 = -M
    # BigM5 = -M
    # BigM6 = -M
    
    #create model
    mip = pe.ConcreteModel()
    
    #define sets
    mip.n = pe.RangeSet(0,env.num_periods-1) 
    mip.n1 = pe.RangeSet(0,env.num_periods)
    mip.m = pe.RangeSet(0,env.num_stages-1) #stages
    mip.m0 = pe.RangeSet(0,env.num_stages-2) #stages (excludes last stage which has no inventory)
    
    #define parameters
    mip.unit_price = pe.Param(mip.m, initialize = {i:env.unit_price[i] for i in mip.m}) #sales price for each stage
    mip.unit_cost = pe.Param(mip.m, initialize = {i:env.unit_cost[i] for i in mip.m}) #purchasing cost for each stage
    mip.demand_cost = pe.Param(mip.m, initialize = {i:env.demand_cost[i] for i in mip.m}) #cost for unfulfilled demand at each stage
    mip.holding_cost = pe.Param(mip.m, initialize = {i:env.holding_cost[i] for i in mip.m}) #inventory holding cost at each stage
    mip.supply_capacity = pe.Param(mip.m0, initialize = {i:env.supply_capacity[i] for i in mip.m0}) #production capacity at each stage
    mip.lead_time = pe.Param(mip.m0, initialize = {i:env.lead_time[i] for i in mip.m0}) #lead times in between stages
    mip.discount = env.discount #time-valued discount 
    backlog = env.backlog #backlog or lost sales
    mip.num_periods = env.num_periods #number of periods
    D = env.demand_dist.rvs(size=env.num_periods,**env.dist_param) #demand profile
    mip.D = pe.Param(mip.n, initialize = {i:D[i] for i in mip.n}) #store demands
    prob = env.demand_dist.pmf(D,**env.dist_param) #probability of each demand based on distribution
    mip.prob = pe.Param(mip.n, initialize = {i:prob[i] for i in mip.n}) #store probability at each period
    
    #define variables
    mip.I = pe.Var(mip.n1,mip.m0,domain=pe.NonNegativeReals) #on hand inventory at each stage
    mip.T = pe.Var(mip.n1,mip.m0,domain=pe.NonNegativeReals) #pipeline inventory in between each stage
    mip.R = pe.Var(mip.n,mip.m0,domain=pe.NonNegativeReals) #reorder quantities for each stage
    # mip.R1 = pe.Var(mip.n,mip.m0,domain=pe.NonNegativeReals) #unconstrained reorder quantity
    mip.S = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals) #sales at each stage
    if backlog:
        mip.B = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals) #backlogs at each stage
    else:
        mip.LS = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals) #lost sales at each stage
    mip.P = pe.Var(mip.n,domain=pe.Reals) #profit at each stage
    # mip.y = pe.Var(mip.n,mip.m0,domain=pe.Binary) #auxiliary variable (y = 0: inventory level is above the base stock level (no reorder))
    # mip.y1 = pe.Var(mip.n,mip.m0,domain=pe.Binary) #auxiliary variable (y1 = 1: unconstrained reorder quantity accepted)
    # mip.y2 = pe.Var(mip.n,mip.m0,domain=pe.Binary) #auxiliary variable (y2 = 1: reorder quantity is capacity constrained)
    # if env.num_stages > 2:
        # mip.y3 = pe.Var(mip.n,mip.m0,domain=pe.Binary) #auxiliary variable (y3 = 1: reorder quantity is inventory constrained)
    # mip.y4 = pe.Var(mip.n,mip.m,domain=pe.Binary) #auxiliary variable (y4 = 1: demand + backlog satisfied)
    # mip.x = pe.Var(mip.m0,domain=pe.NonNegativeReals) #inventory level at each stage
    # mip.z = pe.Var(mip.m0,domain=pe.PositiveIntegers) #base stock level at each stage
    
    #initialize
    for m in mip.m0:
        mip.I[0,m].fix(env.init_inv[m])
        mip.T[0,m].fix(0)
    
    #define constraints
    mip.inv_bal = pe.ConstraintList()
    mip.sales1 = pe.ConstraintList()
    # mip.sales2 = pe.ConstraintList()
    mip.sales3 = pe.ConstraintList()
    # mip.sales4 = pe.ConstraintList()
    mip.sales5 = pe.ConstraintList()
    # mip.reorder1 = pe.ConstraintList()
    # mip.reorder2 = pe.ConstraintList()
    # mip.reorder3 = pe.ConstraintList()
    # mip.reorder4 = pe.ConstraintList()
    # mip.reorder5 = pe.ConstraintList()
    mip.reorder6= pe.ConstraintList()
    # mip.reorder7= pe.ConstraintList()
    mip.reorder8= pe.ConstraintList()
    # mip.reorder9= pe.ConstraintList()
    # mip.reorder10 =  pe.ConstraintList()
    mip.pip_bal = pe.ConstraintList()
    mip.unfulfilled = pe.ConstraintList()
    mip.profit = pe.ConstraintList()
    # mip.basestock = pe.ConstraintList()
    # mip.init_inv= pe.ConstraintList()
    
    #build constraints
    # for m in mip.m0:
        #relate base stock levels to inventory levels: base stock level = total inventory up to that echelon
        # mip.basestock.add(mip.z[m] == sum(mip.x[i] for i in range(m+1)))
        #initialize inventory levels to being full (this would be ideal)
        # mip.init_inv.add(mip.I[0,m] == mip.x[m])
    
    for n in mip.n:
        #calculate profit: apply time value discount to sales revenue - purchasing costs - unfulfilled demand cost - holding cost
        if backlog:
            mip.profit.add(mip.P[n] == mip.discount**n * (sum(mip.unit_price[m]*mip.S[n,m] for m in mip.m)
                                                    - (sum(mip.unit_cost[m]*mip.R[n,m] for m in mip.m0) + mip.unit_cost[mip.m[-1]]*mip.S[n,mip.m[-1]])
                                                    - sum(mip.demand_cost[m]*mip.B[n,m] for m in mip.m)
                                                    - sum(mip.holding_cost[m]*mip.I[n+1,m] for m in mip.m0)))
        else:
            mip.profit.add(mip.P[n] == mip.discount**n * (sum(mip.unit_price[m]*mip.S[n,m] for m in mip.m)
                                                    - (sum(mip.unit_cost[m]*mip.R[n,m] for m in mip.m0) + mip.unit_cost[mip.m[-1]]*mip.S[n,mip.m[-1]])
                                                    - sum(mip.demand_cost[m]*mip.LS[n,m] for m in mip.m)
                                                    - sum(mip.holding_cost[m]*mip.I[n+1,m] for m in mip.m0)))
            
        for m in mip.m0:
            #on-hand inventory balance: next period inventory = prev period inventory + arrival from above stage - sales
            if n - mip.lead_time[m] >= 0:
                mip.inv_bal.add(mip.I[n+1,m] == mip.I[n,m] + mip.R[n - mip.lead_time[m],m] - mip.S[n,m])
            else:
                mip.inv_bal.add(mip.I[n+1,m] == mip.I[n,m] - mip.S[n,m])
            #pipeline inventory balance: next period inventory = prev period inventory - delivered material + new reorder
            if n - mip.lead_time[m] >= 0:
                mip.pip_bal.add(mip.T[n+1,m] == mip.T[n,m] - mip.R[n - mip.lead_time[m],m] + mip.R[n,m])
            else:
                mip.pip_bal.add(mip.T[n+1,m] == mip.T[n,m] + mip.R[n,m])
            #reorder quantity constraints: R = min(c, I[m+1], R1)
                #last constraint ensures that only one of the 3 options is chosen
                # Note: R = min(A,B,C) <-> A + M*(1-y1) <= R <= A  ;  B + M*(1-y2) <= R <= B  ;  C + M*(1-y3) <= R <= C  ;  y1+y2+y3==1
            # mip.reorder4.add(mip.R[n,m] <= mip.R1[n,m])
            # mip.reorder5.add(mip.R[n,m] >= mip.R1[n,m] + BigM3 * (1 - mip.y1[n,m]))
            mip.reorder6.add(mip.R[n,m] <= mip.supply_capacity[m])
            # mip.reorder7.add(mip.R[n,m] >= mip.supply_capacity[m] * mip.y2[n,m])
            if (m < mip.m0[-1]) & (env.num_stages > 2): 
                #if number of stages = 2, then there is no inventory constraint since the last level has unlimited inventory
                #also, last stage has unlimited inventory
                mip.reorder8.add(mip.R[n,m] <= mip.I[n,m+1])
                # mip.reorder9.add(mip.R[n,m] >= mip.I[n,m+1] + BigM4 * (1 - mip.y3[n,m]))
                # mip.reorder10.add(mip.y1[n,m] + mip.y2[n,m] + mip.y3[n,m] == 1)
            # else:
                # mip.reorder10.add(mip.y1[n,m] + mip.y2[n,m] == 1)
                
        for m in mip.m:            
            if m == 0:
            #sales constraints: S = min(I + R[n-L], D + B[n-1]) at stage 0
                if n - mip.lead_time[m] >= 0:
                    mip.sales1.add(mip.S[n,m] <= mip.I[n,m] + mip.R[n - mip.lead_time[m],m])
                    # mip.sales2.add(mip.S[n,m] >= mip.I[n,m] + mip.R[n - mip.lead_time[m],m] + BigM5 * (1 - mip.y4[n,m]))
                else:
                    mip.sales1.add(mip.S[n,m] <= mip.I[n,m])
                    # mip.sales2.add(mip.S[n,m] >= mip.I[n,m] + BigM5 * (1 - mip.y4[n,m]))
                
                if (backlog) & (n-1>=0):
                    mip.sales3.add(mip.S[n,m] <= mip.D[n] + mip.B[n-1,m])
                    # mip.sales4.add(mip.S[n,m] >= mip.D[n] + mip.B[n-1,m] + BigM6 * mip.y4[n,m])
                else:
                    mip.sales3.add(mip.S[n,m] <= mip.D[n])
                    # mip.sales4.add(mip.S[n,m] >= mip.D[n] + BigM6 * mip.y4[n,m])
            else:
            #sales constraints: S = R[n,m-1] at higher level stages
                mip.sales5.add(mip.S[n,m] == mip.R[n,m-1])
                    
            if m == 0:
            #unfulfilled orders at stage 0: U = D + B[n-1] - S
                if backlog:
                    if n-1>=0:
                        mip.unfulfilled.add(mip.B[n,m] == mip.D[n] + mip.B[n-1,m] - mip.S[n,m])
                    else:
                        mip.unfulfilled.add(mip.B[n,m] == mip.D[n] - mip.S[n,m])
                else:
                    mip.unfulfilled.add(mip.LS[n,m] == mip.D[n] - mip.S[n,m])
            else:
            #unfulfilled orders at stage higher level stages: U = R[n,m-1] + B[n-1,m] - S[n,m]
                if backlog:
                    if n-1>=0:
                        mip.unfulfilled.add(mip.B[n,m] == mip.R[n,m-1] + mip.B[n-1,m] - mip.S[n,m])
                    else:
                        mip.unfulfilled.add(mip.B[n,m] == mip.R[n,m-1] - mip.S[n,m])
                else:
                    mip.unfulfilled.add(mip.LS[n,m] == mip.R[n,m-1] - mip.S[n,m])

    #objective function: maximize average profit
    mip.obj = pe.Objective(
        expr = 1/mip.num_periods * sum(mip.P[n] for n in mip.n),
        sense = pe.maximize)
    
    return mip
