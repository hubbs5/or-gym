#!usr/bin/env python

from or_gym.algos import rl_utils
from datetime import datetime
import os

if __name__ == "__main__":
	timestamp = datetime.strftime(datetime.today(), '%Y-%m-%d')
	model_name = 'or_gym_tune'
	env_name = 'Knapsack-v0'
	rl_config = rl_utils.check_config(env_name)
	print('\n')
	_ = [print(k, rl_config[k]) for k in rl_config.keys()]
	results = rl_utils.tune_model(env_name, rl_config)
	if os.path.exists('results') == False:
		os.mkdir('results')

	results.dataframe().to_csv('results/' + env_name + '_' + model_name + timestamp + '.csv')