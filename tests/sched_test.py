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

    def test_book_inventory(self, conf):
        env = self._build_env(conf)
        env.inventory *= 0
        prod_tuple = utils.ProdTuple(
            Stage=0,
            Line=0,
            Number=0,
            ProdStartTime=0,
            ProdReleaseTime=48,
            ProductID=0,
            Quantity=100
        )
        env.env_time = prod_tuple.ProdReleaseTime
        try:
            env._book_inventory(prod_tuple)
        except TypeError:
            # Ignore error from having no schedule in submodule
            pass
        exp_inv = np.zeros(env.inventory.shape)
        exp_inv[prod_tuple.ProductID] += prod_tuple.Quantity
        assert env.inventory[prod_tuple.ProductID] == prod_tuple.Quantity, (
            f"Actual Inventory: {env.inventory}\n" +
            f"Excpected Inventory: {exp_inv}"
        )

    def test_append_schedule(self, conf):
        env = self._build_env(conf)
        action = [1] # Must be iterable
        prod_time = env._min_production_qty / env._run_rate
        off_grade_time = 0
        release_time = prod_time + env._cure_time
        sched = np.zeros(len(env.sched_cols))
        sched[env.sched_col_idx['Number']] += 1
        sched[env.sched_col_idx['Quantity']] += env._min_production_qty
        sched[env.sched_col_idx['ProductID']] += action
        sched[env.sched_col_idx['ProdStartTime']] += env.env_time
        sched[env.sched_col_idx['ProdEndTime']] += env.env_time + prod_time
        sched[env.sched_col_idx['ProdReleaseTime']] += env.env_time + release_time

        env.append_schedules(action)

        assert np.allclose(env.schedules[0][0], sched), ("Schedules don't match:\n" +
            f"Col\t\tActual\tExpected\n" +
            f"{env.sched_cols}\n" +
            f"{env.schedules[0][0]}\n" +
            f"{sched}")

    def test_ship_orders(self, conf):
        env = self._build_env(conf)
        qty = 100
        # Set order book and inventory such that all orders will be shipped
        env.inventory = np.ones(env.inventory.shape) * qty
        order_book = np.zeros((env.n_fin_products, len(env.order_book_cols)))
        order_book[:, env.ob_col_idx['Number']] = np.arange(env.n_fin_products)
        order_book[:, env.ob_col_idx['ProductID']] = env.product_ids
        order_book[:, env.ob_col_idx['CreateTime']] = env.env_time
        order_book[:, env.ob_col_idx['DueTime']] = env.env_time
        order_book[:, env.ob_col_idx['Quantity']] = qty
        env.order_book = order_book.copy()

        env.ship_orders()
        # Ensure final products have all been shipped
        # TODO: Update for multi-stage models
        assert env.inventory.sum() == 0, (
            f"Discrepancy in inventory levels:\n" +
            f"Expected = 0\tActual = {env.inventory.sum()}\n" +
            f"{env.inventory}")

    def test_run_episode(self, conf):
        env = self._build_env(conf)
        limit = 1E6
        c = 0
        while True:
            action = env.action_space.sample()
            s, r, d, _ = env.step(action)
            c += 1
            if d:
                break
            if c > limit:
                break
        assert c < limit