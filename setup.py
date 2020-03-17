#!usr/bin/env python

from setuptools import setup, find_packages
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'or_gym'))
from version import VERSION

setup(name='or-gym',
	version=VERSION,
	description='OR-Gym: A set of environments for developing reinforcement learning agents for OR problems.',
	author='Christian Hubbs',
	license='MIT',
	# pacakges=[package for package in find_packages() if package.startswith('something')],
	install_requires=['gym>=0.15.0','matplotlib>=3.0.3', 'str2bool>=1',
		'numpy>=1.16.1','scipy>=1.4.1','torch>=1.4.0','torchvision>=0.5.0','pyomo>=5.5',
		'ray>=0.8', 'pandas>=1.0'],
	zip_safe=False
)