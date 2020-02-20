import gym

class KnapsackEnv(gym.Env):

    def __init__(self):
        print('initialized')
    def step(self):
        print('step')
    def reset(self):
        print('env reset')
