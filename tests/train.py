#!usr/bin/env python

import or_gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from or_gym.utils.env_config import *
from or_gym.algos.rl_utils import *
import ray
from ray.rllib import agents
from copy import deepcopy
import time
from argparse import ArgumentParser
import pickle

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='PortfolioOpt-v0')
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--plot_results', type=bool, default=True)
    parser.add_argument('--print', type=bool, default=True,
        help='Print output.')
    parser.add_argument('--algo', type=str, default='PPO')

    return parser.parse_args()

def set_config(default_config, config_dict=None):
    config = deepcopy(default_config)
    if type(config_dict) == dict:
        for k in config.keys():
            if k in config_dict.keys():
                if type(config[k]) == dict:
                    for m in config[k].keys():
                        if m in config_dict.keys():
                            config[k][m] = config_dict[m]
                else:
                    config[k] = config_dict[k]
            else:
                continue
                
    return config

def train_agent(env_name, algo='a3c', iters=50, config_dict={}, print_output=True):
    if hasattr(agents, algo):
        agent = getattr(agents, algo)
        config = set_config(agent.DEFAULT_CONFIG, config_dict)
        trainer = getattr(agent, algo.upper() + 'Trainer')(config, 
            env=create_env(env_name))
    else:
        raise AttributeError('No attribute {}'.format(algo))
    t0 = time.time()
    results = []
    for n in range(iters):
        t1 = time.time()
        result = trainer.train()
        t2 = time.time()
        print(result['info']['learner'])
        results.append(result)
        if (n + 1) % 10 == 0 and print_output:
            print("Iter:\t{}\tMean Rewards:\t{:.1f}".format(n+1, result['episode_reward_mean']) + 
                  "\tEps per second:\t{:.3f}\tTotal Time (s):\t{:.1f}".format(
                      result['episodes_this_iter']/(t2-t1), t2-t0))
    # print(results[-1])
    return trainer, results

if __name__ == '__main__':
	ray.init()
	args = parse_arguments()
	# config_dict = {'grad_clip': 10}
	trainer, results = train_agent(args.env, args.algo.lower(), 
		args.iters, config_dict={}, print_output=args.print)
	if args.plot_results:
		mean_rewards = [i['episode_reward_mean'] for i in results]
		try:
			policy_loss = [i['info']['learner']['default_policy']['policy_loss'] for i in results]
			value_loss = [i['info']['learner']['default_policy']['vf_loss'] for i in results]
		except KeyError:
			policy_loss = [i['info']['learner']['policy_loss'] for i in results]
			value_loss = [i['info']['learner']['vf_loss'] for i in results]

		fig, ax = plt.subplots(3, 1, figsize=(12,8))
		ax[0].plot(mean_rewards)
		ax[0].set_title('Rewards')
		ax[1].plot(policy_loss)
		ax[1].set_title('Policy Loss')
		ax[2].plot(value_loss)
		ax[2].set_title('Value Loss')
		plt.tight_layout()
		plt.show()
