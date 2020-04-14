#!usr/bin/env python

from or_gym.algos.newsvendor.math_prog import *
# from or_gym.algos.newsvendor.heuristics import *
from or_gym.algos.math_prog_utils import *
# from or_gym.algos.heuristic_utils import *
import gym
import or_gym
import sys
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--print', type=bool, default=True,
        help='Print output.')

    return parser.parse_args()

def optimize_nv_ip(env):
    env.reset() #reset env
    model = build_nv_ip_model(env)
    model, results = solve_math_program(model, solver = 'gurobi')
    return model, results
    
def optimize_nv_min(env):
    env.reset() #reset env
    results = solve_min_program(env, fun = nv_min_model, local_search = True)
    return results

# if __name__ == '__main__':
    #parser = parse_arguments()
    #args = parser(sys.argv)
    # env = gym.make('Knapsack-v1')
    # model, results = optimize_bkp(env)
    # print("Optimal reward\t\t=\t{}".format(model.obj.expr()))
    # actions, rewards = bkp_heuristic(env)
    # print("Heuristic reward\t=\t{}".format(sum(rewards)))
    #print("RL reward\t\t=\t{}".format())