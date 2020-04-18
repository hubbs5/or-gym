#!usr/bin/env python

from or_gym.algos.newsvendor.math_prog import *
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

def online_optimize_nv_mip(env,solver='gurobi',solver_kwargs={},warmstart=False,warmstart_kwargs={}):
    # raise NotImplementedError('ONV (MIP) optimization not yet implemented.')
    env.reset() #reset env
    
    #initialize
    actions, rewards, basestock = [], [], []
    done = False
    #take a step in the simulation
    action = env.base_stock_action(z=env.init_inv) #don't order anything
    state, reward, done, _ = env.step(action)
    #store results
    actions.append(action)
    rewards.append(reward)
    basestock.append(env.init_inv)
    
    while not done:
        #print period
        print("*******************************************\nPeriod: {} \n".format(env.period)) 
        #build model
        model = build_nv_mip_model(env,online=True) 
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
        try:
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
        basestock.append(zopt)

    return actions, rewards, basestock
    
def online_optimize_nv_dfo(env):
    # raise NotImplementedError('ONV (min) optimization not yet implemented.')
    env.reset() #reset env
    
    #initialize
    actions, rewards, basestock = [], [], []
    done = False
    #take a step in the simulation
    action = env.base_stock_action(z=env.init_inv) #don't order anything
    state, reward, done, _ = env.step(action)
    #store results
    actions.append(action)
    rewards.append(reward)
    basestock.append(env.init_inv)

    while not done:
        #print period
        print("*******************************************\nPeriod: {} \n".format(env.period)) 
        #run DFO
        results = solve_dfo_program(env, fun = nv_dfo_model, online=True, local_search = True)
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
        basestock.append(zopt)
        
    return actions, rewards, basestock

# def optimize_onv(env, print_results=False):
    # model = build_onv_ip_model(env)
    # model, results = solve_math_program(model, print_results=print_results)

    # return model, results

# if __name__ == '__main__':
    #parser = parse_arguments()
    #args = parser(sys.argv)
    # env = gym.make('Knapsack-v2')

    #Keep items constant across all applications
    # N_SCENARIOS = 1000
    # item_sequence = np.random.choice(env.item_numbers, 
        # size=(N_SCENARIOS, env.step_limit), p=env.item_probs)
    # avg_opt_rewards = 0
    # for n in range(N_SCENARIOS):
        # env.reset()
        # model, results = optimize_onv(env, item_sequence[n])
        # avg_opt_rewards += (model.obj.expr() - avg_opt_rewards) / (n + 1)

    # print("Average Optimal Reward\t\t=\t{}".format(avg_opt_rewards))

    # avg_heur_rewards = 0
    # for n in range(N_SCENARIOS):
        # env.reset()
        # actions, items, rewards = okp_heuristic(env, item_sequence[n])
        # avg_heur_rewards += (sum(rewards) - avg_heur_rewards) / (n + 1)
    # print("Average Heuristic Reward\t=\t{:.2f}".format(avg_heur_rewards))
    
    #print("Average RL Reward\t\t=\t{}".format())