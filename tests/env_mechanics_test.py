#!usr/bin/env python

# Imports environment and runs 1,000 episodes with random actions to ensure
# there are no basic errors in the environment.

import gym
import or_gym
from argparse import ArgumentParser
import string

env_list = ['Knapsack-v0', 'Knapsack-v1', 'Knapsack-v2',
            'BinPacking-v0', 'BinPacking-v1', 'BinPacking-v2',
            'VMPacking-v0', 'VMPacking-v1',
            'PortfolioOpt-v0',
            'TSP-v0',
            'VehicleRouting-v0', 'VehicleRouting-v1',
            'NewsVendor-v0', 'NewsVendor-v1']

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='all')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--print', type=bool, default=True,
        help='Print output.')

    return parser.parse_args()

def test_env(env, n_episodes, print_output=True):
	for ep in range(n_episodes):
		env.reset()
		rewards = 0
		done = False
		while done == False:
			action = env.action_space.sample()
			s, r, done, _ = env.step(action)
			rewards += r
			if done and ep % 100 == 0 and print_output:
				print("Ep {}\t\tRewards={}".format(ep, rewards))
	print('Test Complete\n')

if __name__ == "__main__":
	args = parse_arguments()
	env_name = args.env
	n_episodes = args.episodes
	print_output = args.print 

	if env_name.lower() == 'all':
		for name in env_list:
			print('Testing {}'.format(name))
			try:
				env = gym.make(name)
				try:
					test_env(env, n_episodes)
				except Exception as e:
					print('Error encountered\n')
			except Exception as e:
				print('Error initializing {}\n'.format(name))
	else:
		env = gym.make(env_name)
		print('Testing {}'.format(env_name))
		test_env(env, n_episodes)
