#!usr/bin/env python

from or_gym.algos.vm_packing.math_prog import *
from or_gym.algos.vm_packing.heuristics import *
from or_gym.algos.math_prog_utils import *
import or_gym
import sys
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--print', type=bool, default=True,
        help='Print output.')
    parser.add_argument('--solver', type=str, default='glpk')

    return parser.parse_args()

def optimize_vmp_temp(env):
    # Run iteratively to make model tractable
    model, actions, rewards = solve_shrinking_horizon_mp(env, build_online_vm_opt,
        extract_vm_packing_plan, 'gurobi', True)
    return model, actions, rewards

if __name__ == '__main__':
    # parser = parse_arguments()
    # args = parser(sys.argv)
    env_name = 'VMPacking-v0'
    env = or_gym.make(env_name)
    env.step_limit = 11
    env.n_pms = 10
    opt_model, opt_actions, opt_rewards = optimize_vmp_temp(env)
    # heur_actions, heur_rewards = vmp_temp_heuristics(env)
    # print("Testing Trained RL agent...")
    # rl_model, rl_rewards, episodes = train_rl_knapsack(env_name, rl_config)
    print("Optimal reward\t\t=\t{:.1f}".format(sum(opt_rewards)))
    # print("Heuristic reward\t=\t{:.1f}".format(sum(heur_rewards)))
    # print("RL reward\t=\t{:.1f}".format(rl_rewards[-1]))