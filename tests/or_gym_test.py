#!usr/bin/env python

import gym
import or_gym
import sys
from argparse import ArgumentParser

def parse_arguments():
	parser = ArgumentParser()
	parser.add_argument('--env', type=str, default='Knapsack-v0',
		help='Set test environment.')

	return parser.parse_args()

def main(args):
	args = parse_arguments()
	env = gym.make(args.env)
	print('{} initialized successfully'.format(args.env))
	action = env.sample_action()
	print(env.step(1))
	print('Step successful')
	env.reset()
	print('Reset successful')

if __name__ == '__main__':
	main(sys.argv)