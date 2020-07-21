import or_gym
import sys
import traceback

env_base_name = 'BinPacking-v'
successes, errors = 0, 0
for i in range(6):
    env_name = env_base_name + str(i)
    try:
        env = or_gym.make(env_name)
        print("{} built succesfully.".format(env_name))
        successes += 1
    except Exception as e:
        print("\n{} encountered an error.".format(env_name))
        traceback.print_exc()
        errors += 1

print()
print("{} succesful environments".format(successes))
print("{} failed environments".format(errors))