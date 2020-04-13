#!usr/bin/env python

from or_gym.algos.newsvendor.math_prog import *
# from or_gym.algos.newsvendor.heuristics import *
from or_gym.algos.math_prog_utils import *
# from or_gym.algos.heuristic_utils import *
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

def online_optimize_onv_ip(env):
    # raise NotImplementedError('ONV (MIP) optimization not yet implemented.')
    env.reset() #reset env
    
	actions, rewards = [], []
	done = False
	action = env.base_stock_action(z=env.init_inv)
	state, reward, done, _ = env.step(action)
	actions.append(action)
	rewards.append(reward)
	while not done:
		model = build_nv_ip_model(env,online=True)
		model, results = solve_math_program(model,solver='gurobi')
		zopt = list(model.z.get_values().values()) # Extract base stock level
		action = env.base_stock_action(z=zopt) # Extract action
		state, reward, done, _ = env.step(action)
		actions.append(action)
		rewards.append(reward)

	return actions, rewards
    
def online_optimize_onv_min(env):
	# raise NotImplementedError('ONV (min) optimization not yet implemented.')
    env.reset() #reset env
    
	actions, rewards = [], []
	done = False
	action = env.base_stock_action(z=env.init_inv)
	state, reward, done, _ = env.step(action)
	actions.append(action)
	rewards.append(reward)
	while not done:
		results = solve_min_program(env, fun = nv_min_model, online=True, local_search = True)
		print(results)
		# Extract base stock level
		zopt = results.zopt
		# Extract action
		action = env.base_stock_action(z=zopt)
		state, reward, done, _ = env.step(action)
		actions.append(action)
		rewards.append(reward)

	return actions, rewards

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