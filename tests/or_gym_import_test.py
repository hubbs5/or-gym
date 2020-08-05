#!usr/bin/env python

import gym
import or_gym
import sys
from argparse import ArgumentParser
from or_gym.version import ENV_LIST

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Knapsack-v0',
        help='Set test environment.')

    return parser.parse_args()

def main(args):
    args = parse_arguments()
    for env_name in ENV_LIST:
        print('\nTesting functionality for {}'.format(env_name))
        try:
            env = or_gym.make(env_name)
            print('{} initialized successfully'.format(env_name))
            try:
                action = env.action_space.sample()
                print('Action {} selected'.format(action))
            except Exception as e:
                print('Error sampling action for env = {}'.format(env_name))
            try:
                _ = env.step(action)
                print('Step successful')
            except Exception as e:
                print('Error encountered during step for action {}.'.format(action))
            try:
                env.reset()
                print('Reset successful for env = {}'.format(env_name))
            except Exception as e:
                print('Reset error encountered for env = {}'.format(env_name))
        except Exception as e:
            print('Error encountered initializing env = {}'.format(env_name))
        

if __name__ == '__main__':
    main(sys.argv)