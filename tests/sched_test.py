import or_gym
from or_gym import utils
from or_gym.envs.supply_chain.scheduling import BaseSchedEnv
import numpy as np
import pytest
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

# Test scenarios and configurations
s0 = ("SS-no-mask",
    {"conf": {
        "env_name": "Scheduling-v0",
        "env_config": {
            "mask": False
        }
    }
})

s1 = ("SS-mask",
    {"conf": {
        "env_name": "Scheduling-v0",
        "env_config": {
            "mask": True
        }
    }
})

class SchedEnv(BaseSchedEnv):

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        return self._RESET()

    def step(self, action):
        return self._STEP(action)

class TestSchedEnvs:

    scenarios = [s0, s1]
    
    def _build_env(self, conf):
        env_config = conf["env_config"]
        env = or_gym.make(conf["env_name"], env_config=env_config)
        return env

    def test_make(self, conf):
        # Tests building the environment
        try:
            self._build_env(conf)
            success = True
        except Exception as e:
            tb = e.__traceback__
            success = False
        assert success, "".join(traceback.format_tb(tb))

    def test_init_demand_model(self, conf):
        env = SchedEnv()


    def test_book_inventory(self, conf):
        env = self._build_env(conf)
        env.inventory *= 0
        prod_tuple = utils.ProdTuple(
            Stage=0,
            Train=0,
            BatchNumber=0,
            ProdStartTime=0,
            ProdReleaseTime=48,
            ProdID=0,
            BatchSize=100
        )
        env.env_time = prod_tuple.ProdReleaseTime
        env.book_inventory(prod_tuple)
        exp_inv = np.zeros(env.inventory.shape)
        exp_inv[prod_tuple.ProdID] += prod_tuple.BatchSize
        assert env.inventory[prod_tuple.ProdID] == prod_tuple.BatchSize, (
            f"Actual Inventory: {env.inventory}\n" +
            f"Excpected Inventory: {exp_inv}"
        )

    def test_append_schedule(self, conf):
        env = self._build_env(conf)
        action = 1
        