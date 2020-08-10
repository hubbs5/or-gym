#!usr/bin/env python

from setuptools import setup, find_packages
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'or_gym'))
from or_gym.version import VERSION

setup(name='or-gym',
	version=VERSION,
	description='OR-Gym: A set of environments for developing reinforcement learning agents for OR problems.',
	author='Christian Hubbs',
	license='MIT',
	url='https://github.com/hubbs5/or-gym',
	# pacakges=[package for package in find_packages() if package.startswith('something')],
	install_requires=[
		'gym>=0.15.0',
		'numpy>=1.16.1',
		'scipy>=1.4.1'],
	zip_safe=False,
	python_requires='>=3.5'
)