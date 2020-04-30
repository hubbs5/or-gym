from gym.envs.registration import EnvRegistry

registry = EnvRegistry()

def register(id, **kwargs):
    return registry.register(id, **kwargs)

def make(id, **kwargs):
    return registry.make(id, **kwargs)

def spec(id):
    return registry.spec(id)