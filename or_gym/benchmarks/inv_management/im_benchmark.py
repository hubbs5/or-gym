#!usr/bin/env python

from or_gym.algos.inv_management.math_prog import *
from or_gym.algos.math_prog_utils import *
import gym
import or_gym
import sys
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--print', type=bool, default=True,
        help='Print output.')

    return parser.parse_args()

def optimize_im_mip(env,solver='gurobi',solver_kwargs={},warmstart=False,warmstart_kwargs={}):
    #run optimization
    env.reset()
    env.seed(env.seed_int)
    model = build_im_mip_model(env)
    model, results = solve_math_program(model, solver = solver, solver_kwargs = solver_kwargs,
                                        warmstart=warmstart, warmstart_kwargs=warmstart_kwargs)
    zopt = list(model.z.get_values().values()) #extract optimal base stock levels
    
    #reset env to run simulation with base stock levels found
    env.reset() 
    env.seed(env.seed_int)
    #run simulation
    for t in range(env.num_periods):
        #take a step in the simulation using base stock policy
        env.step(action=env.base_stock_action(z=zopt)) 
    return model, results
    
def optimize_im_dfo(env):
    #run optimization
    env.reset()
    env.seed(env.seed_int)
    results = solve_dfo_program(env, fun = im_dfo_model, local_search = True)
    
    #reset env to run simulation with base stock levels found
    env.reset() 
    env.seed(env.seed_int)
    #run simulation
    for t in range(env.num_periods):
        #take a step in the simulation using base stock policy
        env.step(action=env.base_stock_action(z=results.zopt)) 
    return results
    
def optimize_im_pi_mip(env,solver='gurobi',solver_kwargs={},warmstart=False,warmstart_kwargs={}):
    #run optimization
    env.reset()
    env.seed(env.seed_int)
    model = build_im_pi_mip_model(env)
    model, results = solve_math_program(model, solver = solver, solver_kwargs = solver_kwargs,
                                        warmstart=warmstart, warmstart_kwargs=warmstart_kwargs)
    N = env.num_periods
    M = env.num_stages
    Ropt = np.reshape(list(model.R.get_values().values()),(N,M-1)) #extract optimal reorder quantities
    
    #reset env to run simulation with base stock levels found
    env.reset() 
    env.seed(env.seed_int)
    #run simulation
    for t in range(env.num_periods):
        #take a step in the simulation using base stock policy
        env.step(action=Ropt[t,:]) 
    return model, results
