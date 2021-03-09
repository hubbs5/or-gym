import or_gym
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
            "mask": True
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