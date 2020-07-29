#!usr/bin/env python

# Imports environment and runs 1,000 episodes with random actions to ensure
# there are no basic errors in the environment.

import or_gym
from argparse import ArgumentParser
import time
import numpy as np

env_list = ['Knapsack-v0', 'Knapsack-v1', 'Knapsack-v2', 'Knapsack-v3',
            'BinPacking-v0', 'BinPacking-v1', 'BinPacking-v2',
			'BinPacking-v3', 'BinPacking-v4', 'BinPacking-v5',
            'VMPacking-v0', 'VMPacking-v1',
            'PortfolioOpt-v0',
            'TSP-v0',
			'InvManagement-v0', 'InvManagement-v1',
            'Newsvendor-v0']
			#'VehicleRouting-v0']

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='all')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--print', type=bool, default=True,
        help='Print output.')

    return parser.parse_args()

def test_env(env, n_episodes, print_output=True):
	t0 = time.time()
	total_steps, steps = [], []
	for ep in range(n_episodes):
		env.reset()
		rewards = 0
		done = False
		step_count = 0
		while done == False:
			action = env.action_space.sample()
			s, r, done, _ = env.step(action)
			# Check env vals
			valid_state = env.observation_space.contains(s)
			if valid_state == False:
				msg = 'Observation Space does not match:'
				msg += '\nobservation_space:\nShape:\t{}\n\t{}'.format(s.shape, s)
				msg += '\nAction:\t{}'.format(action)
				raise ValueError(msg)
			rewards += r
			step_count += 1
			if done:
				total_steps.append(step_count)
				steps.append(step_count)
				if (ep + 1) % 100 == 0 and print_output:
					print("Ep {}\t\tRewards={:.1f}\tMean Steps={:.1f}\t".format(
						ep + 1, rewards, np.mean(step_count)))
					step_count = []

	t1 = time.time()
	print('Test Complete\t{:.04f}s/100 steps\n'.format((t1-t0)/sum(steps)*100))

if __name__ == "__main__":
	args = parse_arguments()
	env_name = args.env
	n_episodes = args.episodes
	print_output = args.print 

	if env_name.lower() == 'all':
		for name in env_list:
			print('Testing {}'.format(name))
			try:
				env = or_gym.make(name)
				try:
					test_env(env, n_episodes)
				except Exception as e:
					print('Error encountered\n')
			except Exception as e:
				print('Error initializing {}\n'.format(name))
	else:
		env = or_gym.make(env_name)
		print('Testing {}'.format(env_name))
		test_env(env, n_episodes)
