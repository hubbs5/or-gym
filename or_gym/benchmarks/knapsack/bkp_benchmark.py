#!usr/bin/env python

from or_gym.algos.knapsack.math_prog import *
from or_gym.algos.knapsack.heuristics import *
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

def optimize_bkp(env, print_results=False):

    model = build_bkp_ip_model(env)
    model, results = solve_math_program(model, print_results=print_results)
    return model, results

if __name__ == '__main__':
    parser = parse_arguments()
    args = parser(sys.argv)
    env = gym.make('Knapsack-v1')
    model, results = optimize_bkp(env)
    print("Optimal reward\t\t=\t{}".format(model.obj.expr()))
    actions, rewards = bkp_heuristic(env)
    print("Heuristic reward\t=\t{}".format(sum(rewards)))
    print("RL reward\t\t=\t{}".format())