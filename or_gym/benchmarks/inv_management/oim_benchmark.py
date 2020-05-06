#!usr/bin/env python

from or_gym.algos.inv_management.math_prog import *
from or_gym.algos.math_prog_utils import *
import gym
import or_gym
import numpy as np
import sys
from argparse import ArgumentParser
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition

np.random.seed(0)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--print', type=bool, default=True,
        help='Print output.')

    return parser.parse_args()

def online_optimize_im_mip(env,solver='gurobi',solver_kwargs={},warmstart=False,warmstart_kwargs={}):
    # raise NotImplementedError('ONV (MIP) optimization not yet implemented.')
    env.reset() #reset env
    env.seed(env.seed_int)
    
    #initialize
    actions, rewards, basestock = [], [], []
    done = False
    #take a step in the simulation
    action = env.base_stock_action(z=env.init_inv) #don't order anything
    state, reward, done, _ = env.step(action)
    #store results
    actions.append(action)
    rewards.append(reward)
    basestock.append(list(env.init_inv))
    
    while not done:
        #print period
        print("*******************************************\nPeriod: {} \n".format(env.period)) 
        #build model
        model = build_im_mip_model(env,online=True) 
        try:
            #solve model
            if warmstart & (len(basestock) > 0):
                model, results = solve_math_program(model,solver=solver,solver_kwargs=solver_kwargs,
                                                    warmstart=True,
                                                    warmstart_kwargs={'mapping_env':env,
                                                                      'mapping_z':basestock[-1],
                                                                      'online':True})
            else:
                model, results = solve_math_program(model,solver=solver,solver_kwargs=solver_kwargs)
            #Extract base stock level
            zopt = list(model.z.get_values().values()) 
        except:
            zopt = basestock[-1]
        #Extract action
        action = env.base_stock_action(z=zopt) 
        #Take a step in the simulation
        state, reward, done, _ = env.step(action)
        #store results
        actions.append(action)
        rewards.append(reward)
        basestock.append(list(zopt))

    return actions, rewards, np.array(basestock)
    
def online_optimize_im_dfo(env):
    # raise NotImplementedError('ONV (min) optimization not yet implemented.')
    env.reset() #reset env
    env.seed(env.seed_int)
    
    #initialize
    actions, rewards, basestock = [], [], []
    done = False
    #take a step in the simulation
    action = env.base_stock_action(z=env.init_inv) #don't order anything
    state, reward, done, _ = env.step(action)
    #store results
    actions.append(action)
    rewards.append(reward)
    basestock.append(list(env.init_inv))

    while not done:
        #print period
        print("*******************************************\nPeriod: {} \n".format(env.period)) 
        #run DFO
        results = solve_dfo_program(env, fun = im_dfo_model, online=True, local_search = True)
        #Extract base stock level
        if results.success:
            zopt = results.zopt
        else:
            zopt = basestock[-1]
        #Extract action
        action = env.base_stock_action(z=zopt)
        #Take a step in the simulation
        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        basestock.append(list(zopt))
        
    return actions, rewards, np.array(basestock)
    