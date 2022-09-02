#!usr/bin/env python

'''
Tests to ensure environments are compatible with RLLib.
Note RLLib is NOT a required package, but tests are included
because it is very useful for RL work.
'''

import or_gym
from or_gym.envs.env_list import ENV_LIST
import ray
from ray.rllib import agents
import traceback

def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")
    

class TestEnv:
  scenarios = [(i, {"config": {"env_name": i}}) for i in ENV_LIST]
  
  def _build_env(self, env_name):
    env = or_gym.make(env_name)
    