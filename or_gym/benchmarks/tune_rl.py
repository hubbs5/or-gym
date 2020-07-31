#!usr/bin/env python

from or_gym.algos import rl_utils
from datetime import datetime
import os

from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Knapsack-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='or_gym_tune')
    parser.add_argument('--algo', type=str, default='PPO')
    parser.add_argument('--print', type=bool, default=True,
        help='Print output.')

    return parser.parse_args()

if __name__ == "__main__":
	args = parse_arguments()
	env_name = args.env
	algo = args.algo.upper()
	model_name = args.model_name
	timestamp = datetime.strftime(datetime.today(), '%Y-%m-%d')
	rl_config = rl_utils.check_config(env_name)
	print('\n')
	_ = [print(k, rl_config[k]) for k in rl_config.keys()]
	results = rl_utils.tune_model(env_name, rl_config, model_name=model_name, algo=algo)
	if os.path.exists('results') == False:
		os.mkdir('results')

	results.dataframe().to_csv('results/' + env_name + '_' + model_name + '_' + timestamp + '.csv',
		index=False)