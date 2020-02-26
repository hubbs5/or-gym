#!usr/bin/env python

from or_gym.algos.knapsack import ukp_math_prog
from or_gym.algos.math_prog_utils import *
import sys
from argparse import ArgumentParser

def parse_arguments():
	parser = ArgumentParser()
	parser.add_argument('--print', type=bool, default=True,
		help='Print output.')

	return parser.parse_args()

def optimize_ukp():

	model = ukp_math_prog.build_ip_model('Knapsack-v0')
	model, results = solve_math_program(model)
	return model, results

if __name__ == '__main__':
	# parser = parse_arguments()
	# args = parser(sys.argv)
	model, results = optimize_ukp()
	print("Total reward = {}".format(model.obj.expr()))