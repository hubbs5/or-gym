#!usr/bin/env python

import gym
import or_gym
import pyomo.environ as pe
import numpy as np

def build_nv_ip_model(env):
    '''
    Optimize base stock level (z variable) on a simulated sample path using an MILP. 
    
    Note: Using the hull reformulation instead of big-M could speed things up. Using tighter 
    big-M could also be helpful.
    
    NOTE: All parameters to the simulation environment must have been defined 
    previously when making the environment.
    
    env = [NewsVendorEnv] current simulation environment. 
    ''' 
    
    assert env.spec.id == 'NewsVendor-v1', \
        '{} received. Heuristic designed for NewsVendor-v1.'.format(env.spec.id)
    #do not reset environment
    
    #big m values
    M = 10000
    BigM1 = M
    BigM2 = M
    BigM3 = -M
    BigM4 = -M
    BigM5 = -M
    BigM6 = -M
    eps = 0.001 #for strict inequalities
    
    #create model
    mip = pe.ConcreteModel()
    
    #define sets
    mip.n = pe.RangeSet(0,env.num_periods-1) 
    mip.n1 = pe.RangeSet(0,env.num_periods)
    mip.m = pe.RangeSet(0,env.num_stages-1)
    mip.m0 = pe.RangeSet(0,env.num_stages-2)
    
    #define parameters
#     mip.init_inv = pe.Param(mip.m, initialize = {i:env.init_inv[i] for i in mip.m0})
    mip.unit_price = pe.Param(mip.m, initialize = {i:env.unit_price[i] for i in mip.m})
    mip.unit_cost = pe.Param(mip.m, initialize = {i:env.unit_cost[i] for i in mip.m})
    mip.demand_cost = pe.Param(mip.m, initialize = {i:env.demand_cost[i] for i in mip.m})
    mip.holding_cost = pe.Param(mip.m, initialize = {i:env.holding_cost[i] for i in mip.m})
    mip.supply_capacity = pe.Param(mip.m0, initialize = {i:env.supply_capacity[i] for i in mip.m0})
    mip.lead_time = pe.Param(mip.m0, initialize = {i:env.lead_time[i] for i in mip.m0})
    mip.discount = env.discount
    mip.num_periods = env.num_periods
    backlog = env.backlog
    D = env.demand_dist.rvs(size=env.num_periods,**env.dist_param)
    mip.D = pe.Param(mip.n, initialize = {i:D[i] for i in mip.n}) #sample demands
    prob = env.demand_dist.pmf(D,**env.dist_param)
    mip.prob = pe.Param(mip.n, initialize = {i:prob[i] for i in mip.n}) #probability at each period
    
    #define variables
    mip.I = pe.Var(mip.n1,mip.m0,domain=pe.NonNegativeReals)
    mip.T = pe.Var(mip.n1,mip.m0,domain=pe.NonNegativeReals)
    mip.R = pe.Var(mip.n,mip.m0,domain=pe.NonNegativeReals)
    mip.R1 = pe.Var(mip.n,mip.m0,domain=pe.NonNegativeReals)
    mip.S = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals)
    if backlog:
        mip.B = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals)
    else:
        mip.LS = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals)
    mip.P = pe.Var(mip.n,domain=pe.Reals)
    mip.y = pe.Var(mip.n,mip.m0,domain=pe.Binary)
    mip.y1 = pe.Var(mip.n,mip.m0,domain=pe.Binary)
    mip.y2 = pe.Var(mip.n,mip.m0,domain=pe.Binary)
    if env.num_stages > 2:
        mip.y3 = pe.Var(mip.n,mip.m0,domain=pe.Binary)
    mip.y4 = pe.Var(mip.n,mip.m,domain=pe.Binary)
    mip.x = pe.Var(mip.m0,domain=pe.PositiveIntegers)
    mip.z = pe.Var(mip.m0,domain=pe.PositiveIntegers)
    
    #initialize
    for m in mip.m0:
#         mip.I[0,m] = mip.init_inv[m]
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
        #relate base stock levels to inventory levels
        mip.basestock.add(mip.z[m] == sum(mip.x[i] for i in range(m+1)))
        #initialize inventory levels to being full
        mip.init_inv.add(mip.I[0,m] == mip.x[m])
    
    for n in mip.n:
        #calculate profit
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
            #on-hand inventory balance
            if n - mip.lead_time[m] >= 0:
                mip.inv_bal.add(mip.I[n+1,m] == mip.I[n,m] + mip.R[n - mip.lead_time[m],m] - mip.S[n,m])
            else:
                mip.inv_bal.add(mip.I[n+1,m] == mip.I[n,m] - mip.S[n,m])
            #pipeline inventory balance
            if n - mip.lead_time[m] >= 0:
                mip.pip_bal.add(mip.T[n+1,m] == mip.T[n,m] - mip.R[n - mip.lead_time[m],m] + mip.R[n,m])
            else:
                mip.pip_bal.add(mip.T[n+1,m] == mip.T[n,m] + mip.R[n,m])
            #reorder quantity constraints: max(0, reorder based on base_stock level)
            if (backlog) & (n-1>=0):
                mip.reorder1.add(mip.R1[n,m] <= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] - mip.B[n-1,i] for i in range(m+1)) + BigM1 * (1 - mip.y[n,m]))
                mip.reorder2.add(mip.R1[n,m] >= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] - mip.B[n-1,i] for i in range(m+1)))
            else:
                mip.reorder1.add(mip.R1[n,m] <= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] for i in range(m+1)) + BigM1 * (1 - mip.y[n,m]))
                mip.reorder2.add(mip.R1[n,m] >= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] for i in range(m+1)))
            mip.reorder3.add(mip.R1[n,m] <= BigM2 * mip.y[n,m])
            #reorder quantity constraints: R = min(c, I[m+1], R1)
                #last constraint ensures that only one of the 3 options is chosen
            mip.reorder4.add(mip.R[n,m] <= mip.R1[n,m])
            mip.reorder5.add(mip.R[n,m] >= mip.R1[n,m] + BigM3 * (1 - mip.y1[n,m]))
            mip.reorder6.add(mip.R[n,m] <= mip.supply_capacity[m])
            mip.reorder7.add(mip.R[n,m] >= mip.supply_capacity[m] * mip.y2[n,m])
            if (m < mip.m0[-1]) & (env.num_stages > 2):
                mip.reorder8.add(mip.R[n,m] <= mip.I[n,m+1])
                mip.reorder9.add(mip.R[n,m] >= mip.I[n,m+1] + BigM4 * (1 - mip.y3[n,m]))
                mip.reorder10.add(mip.y1[n,m] + mip.y2[n,m] + mip.y3[n,m] == 1)
            else:
                mip.reorder10.add(mip.y1[n,m] + mip.y2[n,m] == 1)
                
        for m in mip.m:            
            if m == 0:
            #sales constraints: S = min(I,D) at stage 0
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
            #sales constraints: S = R[n,m-1] + B[n-1,m] at higher level stages
                if (backlog) & (n-1>=0):
                    mip.sales5.add(mip.S[n,m] == mip.R[n,m-1] + mip.B[n-1,m])
                else:
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
        expr = 1/mip.num_periods * sum(mip.P[n]*mip.prob[n] for n in mip.n),
        sense = pe.maximize)
    
    return mip

def build_onv_ip_model(env):
    '''
    Optimize base stock level (z variable) on the existing sample path using an MILP (online). 
    Model takes initial inventory and past demands. It finds the optimal base stock level for
    the current and all previous time points. Note: z is constant in time.
    
    Note: Using the hull reformulation instead of big-M could speed things up. Using tighter 
    big-M could also be helpful.
    
    NOTE: All parameters to the simulation environment must have been defined 
    previously when making the environment.
    
    env = [NewsVendorEnv] current simulation environment. 
    ''' 
    
    assert env.spec.id == 'NewsVendor-v1', \
        '{} received. Heuristic designed for NewsVendor-v1.'.format(env.spec.id)
    #do not reset environment
    
    #big m values
    M = 10000
    BigM1 = M
    BigM2 = M
    BigM3 = -M
    BigM4 = -M
    BigM5 = -M
    BigM6 = -M
    eps = 0.001 #for strict inequalities
    
    #create model
    mip = pe.ConcreteModel()
    
    #define sets
    mip.n = pe.RangeSet(0,env.period-1) 
    mip.n1 = pe.RangeSet(0,env.period)
    mip.m = pe.RangeSet(0,env.num_stages-1)
    mip.m0 = pe.RangeSet(0,env.num_stages-2)
    
    #define parameters
    mip.init_inv = pe.Param(mip.m, initialize = {i:env.init_inv[i] for i in mip.m0})
    mip.unit_price = pe.Param(mip.m, initialize = {i:env.unit_price[i] for i in mip.m})
    mip.unit_cost = pe.Param(mip.m, initialize = {i:env.unit_cost[i] for i in mip.m})
    mip.demand_cost = pe.Param(mip.m, initialize = {i:env.demand_cost[i] for i in mip.m})
    mip.holding_cost = pe.Param(mip.m, initialize = {i:env.holding_cost[i] for i in mip.m})
    mip.supply_capacity = pe.Param(mip.m0, initialize = {i:env.supply_capacity[i] for i in mip.m0})
    mip.lead_time = pe.Param(mip.m0, initialize = {i:env.lead_time[i] for i in mip.m0})
    mip.discount = env.discount
    mip.num_periods = env.period
    backlog = env.backlog
    D = env.D[:env.period]
    mip.D = pe.Param(mip.n, initialize = {i:D[i] for i in mip.n}) #sample demands
    prob = env.demand_dist.pmf(D,**env.dist_param)
    mip.prob = pe.Param(mip.n, initialize = {i:prob[i] for i in mip.n}) #probability at each period
    
    #define variables
    mip.I = pe.Var(mip.n1,mip.m0,domain=pe.NonNegativeReals)
    mip.T = pe.Var(mip.n1,mip.m0,domain=pe.NonNegativeReals)
    mip.R = pe.Var(mip.n,mip.m0,domain=pe.NonNegativeReals)
    mip.R1 = pe.Var(mip.n,mip.m0,domain=pe.NonNegativeReals)
    mip.S = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals)
    if backlog:
        mip.B = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals)
    else:
        mip.LS = pe.Var(mip.n,mip.m,domain=pe.NonNegativeReals)
    mip.P = pe.Var(mip.n,domain=pe.Reals)
    mip.y = pe.Var(mip.n,mip.m0,domain=pe.Binary)
    mip.y1 = pe.Var(mip.n,mip.m0,domain=pe.Binary)
    mip.y2 = pe.Var(mip.n,mip.m0,domain=pe.Binary)
    if env.num_stages > 2:
        mip.y3 = pe.Var(mip.n,mip.m0,domain=pe.Binary)
    mip.y4 = pe.Var(mip.n,mip.m,domain=pe.Binary)
    mip.x = pe.Var(mip.m0,domain=pe.PositiveIntegers)
    mip.z = pe.Var(mip.m0,domain=pe.PositiveIntegers)
    
    #initialize
    for m in mip.m0:
        mip.I[0,m] = mip.init_inv[m]
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
#     mip.init_inv= pe.ConstraintList()
    
    #build constraints
    for m in mip.m0:
        #relate base stock levels to inventory levels
        mip.basestock.add(mip.z[m] == sum(mip.x[i] for i in range(m+1)))
        #initialize inventory levels to being full
#         mip.init_inv.add(mip.I[0,m] == mip.x[m])
    
    for n in mip.n:
        #calculate profit
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
            #on-hand inventory balance
            if n - mip.lead_time[m] >= 0:
                mip.inv_bal.add(mip.I[n+1,m] == mip.I[n,m] + mip.R[n - mip.lead_time[m],m] - mip.S[n,m])
            else:
                mip.inv_bal.add(mip.I[n+1,m] == mip.I[n,m] - mip.S[n,m])
            #pipeline inventory balance
            if n - mip.lead_time[m] >= 0:
                mip.pip_bal.add(mip.T[n+1,m] == mip.T[n,m] - mip.R[n - mip.lead_time[m],m] + mip.R[n,m])
            else:
                mip.pip_bal.add(mip.T[n+1,m] == mip.T[n,m] + mip.R[n,m])
            #reorder quantity constraints: max(0, reorder based on base_stock level)
            if (backlog) & (n-1>=0):
                mip.reorder1.add(mip.R1[n,m] <= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] - mip.B[n-1,i] for i in range(m+1)) + BigM1 * (1 - mip.y[n,m]))
                mip.reorder2.add(mip.R1[n,m] >= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] - mip.B[n-1,i] for i in range(m+1)))
            else:
                mip.reorder1.add(mip.R1[n,m] <= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] for i in range(m+1)) + BigM1 * (1 - mip.y[n,m]))
                mip.reorder2.add(mip.R1[n,m] >= mip.z[m] - sum(mip.I[n,i] + mip.T[n,i] for i in range(m+1)))
            mip.reorder3.add(mip.R1[n,m] <= BigM2 * mip.y[n,m])
            #reorder quantity constraints: R = min(c, I[m+1], R1)
                #last constraint ensures that only one of the 3 options is chosen
            mip.reorder4.add(mip.R[n,m] <= mip.R1[n,m])
            mip.reorder5.add(mip.R[n,m] >= mip.R1[n,m] + BigM3 * (1 - mip.y1[n,m]))
            mip.reorder6.add(mip.R[n,m] <= mip.supply_capacity[m])
            mip.reorder7.add(mip.R[n,m] >= mip.supply_capacity[m] * mip.y2[n,m])
            if (m < mip.m0[-1]) & (env.num_stages > 2):
                mip.reorder8.add(mip.R[n,m] <= mip.I[n,m+1])
                mip.reorder9.add(mip.R[n,m] >= mip.I[n,m+1] + BigM4 * (1 - mip.y3[n,m]))
                mip.reorder10.add(mip.y1[n,m] + mip.y2[n,m] + mip.y3[n,m] == 1)
            else:
                mip.reorder10.add(mip.y1[n,m] + mip.y2[n,m] == 1)
                
        for m in mip.m:            
            if m == 0:
            #sales constraints: S = min(I,D) at stage 0
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
            #sales constraints: S = R[n,m-1] + B[n-1,m] at higher level stages
                if (backlog) & (n-1>=0):
                    mip.sales5.add(mip.S[n,m] == mip.R[n,m-1] + mip.B[n-1,m])
                else:
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
        expr = 1/mip.num_periods * sum(mip.P[n]*mip.prob[n] for n in mip.n),
        sense = pe.maximize)
    
    return mip