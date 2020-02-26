import gym
import or_gym

env = gym.make('Knapsack-v0')
print('Knapsack-v0 initialized successfully')
print(env.step(1))
print('Step successful')
env.reset()
print('Reset successful')
