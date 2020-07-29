#!usr/bin/env python

# Imports environment and runs 1,000 episodes with ray PPO to test
# the functionality. Running with 'all' as env argument will test
# all of the environments.

import gym
import or_gym
from or_gym.algos.rl_utils import *
from ray.rllib.agents import ppo
from argparse import ArgumentParser
import time

env_list = ['Knapsack-v0', 'Knapsack-v1', 'Knapsack-v2', 'Knapsack-v3',
            'BinPacking-v0', 'BinPacking-v1', 'BinPacking-v2',
			'BinPacking-v3', 'BinPacking-v4', 'BinPacking-v5',
            'VMPacking-v0', 'VMPacking-v1',
            'PortfolioOpt-v0',
            'TSP-v0',
			'InvManagement-v0', 'InvManagement-v1',
            'NewsVendor-v0',
			'VehicleRouting-v0']

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='all')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--print', type=bool, default=True,
        help='Print output.')

    return parser.parse_args()

def test_env(env_name, n_episodes, print_output=True):
    trainer = ppo.PPOTrainer(env=create_env(env_name),
        config={
        "env_config": {
            "mask": True
            },
        "vf_share_layers": True,
        "vf_clip_param": 10000, # Set to high number to avoid any warnings
        "model": {
            "fcnet_activation": "elu",
            "fcnet_hiddens": [128, 128, 128]}
        })
    rewards, eps, eps_total = [], [], []
    training = True
    batch = 0
    t_start = time.time()
    while training:
        t_batch = time.time()
        results = trainer.train()
        rewards.append(results['episode_reward_mean'])
        eps.append(results['episodes_this_iter'])
        eps_total.append(results['episodes_total'])
        batch += 1
        t_end = time.time()
        if sum(eps) >= n_episodes:
            training = False
            break
        if batch % 10 == 0 and print_output:
            t = t_end - t_batch
            t_tot = t_end - t_start
            print("\rEpisode: {}\tMean Rewards: {:.1f}\tEpisodes/sec: {:.2f}\tTotal Time: {:.1f}s".format(
                eps_total[-1], rewards[-1], eps[-1]/t, t_tot), end="")
            
    print("Total Training Time: {:.1f}s\t".format(t_end - t_start))

if __name__ == "__main__":
    args = parse_arguments()
    env_name = args.env
    n_episodes = args.episodes
    print_output = args.print
    ray.init(ignore_reinit_error=True)
    if env_name.lower() == 'all':
        for name in env_list:
            print('Testing {}'.format(name))
            try:
                env = gym.make(name)
                try:
                    test_env(name, n_episodes)
                except Exception as e:
                    print('Error encountered\n')
            except Exception as e:
                print('Error initializing {}\n'.format(name))    
    else:
        # env = gym.make(env_name)
        print('Testing {}'.format(env_name))
        test_env(env_name, n_episodes)
