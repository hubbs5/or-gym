#!usr/bin/env python

from or_gym.algos.knapsack.math_prog import *
from or_gym.algos.knapsack.heuristics import *
from or_gym.algos.math_prog_utils import *
import gym
import or_gym
import numpy as np
import sys
from argparse import ArgumentParser

np.random.seed(0)

def parse_arguments():
	parser = ArgumentParser()
	parser.add_argument('--print', type=bool, default=True,
		help='Print output.')

	return parser.parse_args()

def optimize_okp(env, scenario=None):
	raise NotImplementedError('OKP optimization not yet implemented.')

	actions, items, rewards = [], [], []
	done = False
	count = 0
	while not done:
		model = build_okp_ip_model(env, count, scenario)
		model, results = solve_math_program(model)
		# Extract action
		action = get_action(model)
		state, reward, done, _ = env.step(action)
		actions.append(action)
		items.append(env.current_item)
		rewards.append(reward)

	return actions, items, rewards

if __name__ == '__main__':
	# parser = parse_arguments()
	# args = parser(sys.argv)
	env = gym.make('Knapsack-v2')

	# Keep items constant across all applications
	N_SCENARIOS = 1000
	item_sequence = np.random.choice(env.item_numbers, 
		size=(N_SCENARIOS, env.step_limit), p=env.item_probs)
	# avg_opt_rewards = 0
	# for n in range(N_SCENARIOS):
	# 	env.reset()
	# 	actions, items, rewards = optimize_okp(env, item_sequence[n])
	# 	avg_opt_rewards += (sum(rewards) - avg_opt_rewards) / (n + 1)

	# print("Average Optimal Reward\t\t=\t{}".format(avg_opt_rewards))

	avg_heur_rewards = 0
	for n in range(N_SCENARIOS):
		env.reset()
		actions, items, rewards = okp_heuristic(env, item_sequence[n])
		avg_heur_rewards += (sum(rewards) - avg_heur_rewards) / (n + 1)
	print("Average Heuristic Reward\t=\t{:.2f}".format(avg_heur_rewards))
	
	# print("Average RL Reward\t\t=\t{}".format())