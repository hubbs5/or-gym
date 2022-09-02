#!usr/bin/env python

'''
Tests to ensure environments are compatible with RLLib.
Note RLLib is NOT a required package, but tests are included
because it is very useful for RL work.
'''

import or_gym
from or_gym.utils import create_env
from or_gym.envs.env_list import ENV_LIST
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
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
    

def register_env(env_name, env_config={}):
  env = create_env(env_name)
  tune.register_env(env_name,
                    lambda env_name: env(env_name, env_config=env_config))

class TestEnv:
  scenarios = [(i, {"config": {"env_name": i}}) for i in ENV_LIST]
  
  def _build_env(self, env_name):
    env = or_gym.make(env_name)
    return env
  
  def _get_rl_config_dict(self, env_name, env_config={}):
    rl_config = dict(
      env=env_name,
      num_workers=2,
      env_config=env_config,
      model=dict(
          vf_share_layers=False,
          fcnet_activation='elu',
          fcnet_hiddens=[256, 256]
          ),
      lr=1e-5
      )
    return rl_config
  
  def test_ray(self, config):
    env_name = config["env_name"]
    env = self._build_env(env_name)
    register_env(env_name)
    rl_config = self._get_rl_config_dict(env_name)
    ray.init(ignore_reinit_error=True)
    agent = PPOTrainer(env=env_name, config=rl_config)
    # Train 1 episode for testing
    try:
      res = agent.train()
      passed = True
    except:
      passed = False
      
    ray.shutdown()
    assert passed
      
    
    