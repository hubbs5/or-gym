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
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()

def optimize_bin_kp(env, print_results=False):

    model = build_bin_ip_model(env)
    model, results = solve_math_program(model, print_results=print_results)
    return model, results

if __name__ == '__main__':
    args = parse_arguments()
    env_config = {'seed': args.seed}
    env = or_gym.make('Knapsack-v1', env_config=env_config)
    model, results = optimize_bin_kp(env)
    print("Optimal reward\t\t=\t{}".format(model.obj.expr()))
    
    total_heur_rewards = []
    for i in range(args.iters):
        actions, rewards = bkp_heuristic(env)
        total_heur_rewards.append(sum(rewards))
    print("Heuristic reward\t=\t{}\tStd Dev\t=\t{:.1f}".format(np.mean(total_heur_rewards), 
        np.std(total_heur_rewards)))
    # print("RL reward\t\t=\t{}".format())