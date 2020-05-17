#!usr/bin/env python

from or_gym.algos.knapsack.math_prog import *
from or_gym.algos.knapsack.heuristics import *
from or_gym.algos.knapsack.rl import *
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

def optimize_ukp(env):

    model = build_ukp_ip_model(env)
    model, results = solve_math_program(model)
    return model, results

if __name__ == '__main__':
    args = parse_arguments()
    env_name = 'Knapsack-v0'
    env_config = {'seed': args.seed}
    env = or_gym.make(env_name, env_config=env_config)
    rl_config = rl_utils.check_config(env_name)
    opt_model, opt_results = optimize_ukp(env)
    total_heur_rewards = []
    for i in range(args.iters):
        heur_actions, heur_rewards = ukp_heuristic(env)
        total_heur_rewards.append(sum(heur_rewards))
    print("Training RL agent...")
    # rl_model, rl_rewards, episodes = train_rl_knapsack(env_name, rl_config)
    print("Optimal reward\t\t=\t{:.1f}".format(opt_model.obj.expr()))
    print("Heuristic reward\t=\t{:.1f}\tStd\t=\t{:.1f}".format(
        np.mean(total_heur_rewards), np.std(total_heur_rewards)))
    # print("RL reward\t=\t{:.1f}".format(rl_rewards[-1]))