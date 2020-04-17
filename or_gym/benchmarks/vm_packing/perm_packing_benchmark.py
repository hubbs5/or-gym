#!usr/bin/env python

from or_gym.algos.vm_packing.math_prog import *
from or_gym.algos.vm_packing.heuristics import *
from or_gym.algos.math_prog_utils import *
import or_gym
import sys
from argparse import ArgumentParser
from str2bool import str2bool
import re

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--print', type=str2bool, default=True,
        help='Print output.')
    parser.add_argument('--solver', type=str, default='glpk')
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_known_args()

def optimize_vmp_perm(env, solver='glpk', print_output=True):
    # Run iteratively to make model tractable
    model, actions, rewards = solve_shrinking_horizon_mp(env, build_online_vm_opt,
        extract_vm_packing_plan, solver, print_output)
    return model, actions, rewards

if __name__ == '__main__':
    args, unknown = parse_arguments()
    env_name = 'VMPacking-v0'
    env_config = {re.sub('--', '', unknown[i]): unknown[i+1] 
        for i in range(len(unknown)) if i % 2 == 0}
    env_config.update(args.__dict__)
    print(env_config)
    env = or_gym.make(env_name, env_config=env_config)
    # opt_model, opt_actions, opt_rewards = optimize_vmp_perm(env, solver=args.solver, print_output=args.print)
    heur_actions, heur_rewards = first_fit_heuristic(env)
    # print("Testing Trained RL agent...")
    # rl_model, rl_rewards, episodes = train_rl_knapsack(env_name, rl_config)
    # print("Optimal reward\t\t=\t{:.1f}".format(sum(opt_rewards)))
    print("Heuristic reward\t=\t{:.1f}".format(sum(heur_rewards)))
    # print("RL reward\t=\t{:.1f}".format(rl_rewards[-1]))