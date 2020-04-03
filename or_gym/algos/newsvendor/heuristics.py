#!usr/bin/env python

import gym
import or_gym
import numpy as np

def nv_min_model(x,env):
    '''
    Compute negative of the expected profit for a sample path (since we will be using minimization algo).
    
    x = [integer list; dimension |Stages| - 1] total inventory levels at each node.
    env = [NewsVendorEnv] current simulation environment.
    '''
    
    assert env.spec.id == 'NewsVendor-v1', \
        '{} received. Heuristic designed for NewsVendor-v1.'.format(env.spec.id)
    env.reset() #reset environment
    
    x = np.array(x) #inventory level at each node
    z = np.cumsum(x) #base stock levels
    
#     env = env[0]
    m = env.num_stages
    try:
        dimz = len(z)
    except:
        dimz = 1
    assert dimz == m-1, "Wrong dimension on base stock level vector. Should be #Stages - 1."
    
    #run simulation
    for t in range(env.num_periods):
        #take a step in the simulation using critical ratio base stock
        env.step(action=env.base_stock_action(z=z)) 
    
    #probability from past demands (from env)
    prob = env.demand_dist.pmf(env.D,**env.dist_param) 
    
    #expected profit
    return -1/env.num_periods*np.sum(prob*env.P)

def onv_min_model(x,env):
    '''
    Compute negative of the expected profit for a sample path ONLINE (since we will be using minimization algo).
    
    x = [integer list; dimension |Stages| - 1] total inventory levels at each node.
    env = [NewsVendorEnv] current simulation environment.
    '''
    
    assert env.spec.id == 'NewsVendor-v1', \
        '{} received. Heuristic designed for NewsVendor-v1.'.format(env.spec.id)
    #do not reset environment
    
    x = np.array(x) #inventory level at each node
    z = np.cumsum(x) #base stock levels
    
#     env = env[0]
    m = env.num_stages
    try:
        dimz = len(z)
    except:
        dimz = 1
    assert dimz == m-1, "Wrong dimension on base stock level vector. Should be #Stages - 1."
    
    #extract args to pass to re-simulation
    sim_kwargs = {'periods': env.period,
                  'I0': env.I0,
                  'p': env.p,
                  'r': env.r,
                  'k': env.k,
                  'h': env.h,
                  'c': env.c,
                  'L': env.L,
                  'backlog': env.backlog,
                  'dist': 5,
                  'alpha': env.alpha,
                  'user_D': env.D[:env.period]}
    
    #build simulation environment
    sim = or_gym.make("NewsVendor-v1",env_config=sim_kwargs) 
    
    #run simulation
    for t in range(sim.num_periods):
        #take a step in the simulation using critical ratio base stock
        sim.step(action=sim.base_stock_action(z=z)) 
    
    #probability from past demands (from env)
    prob = env.demand_dist.pmf(sim.D,**env.dist_param) 
    
    #expected profit
    return -1/sim.num_periods*np.sum(prob*sim.P)