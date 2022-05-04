# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 21:51:45 2021

@author: phili
"""

import ray
import time

# Start the ray core
ray.init()

# By adding the `@ray.remote` decorator, a regular Python function
# becomes a Ray remote function.
@ray.remote
def f(x):
    return x * x

# To invoke this remote function, use the `remote` method.
# This will immediately return an object ref (a future) and then create
# a task that will be executed on a worker process.
futures = [f.remote(i) for i in range(4)]

# The result can be retrieved with ``ray.get``.
print(ray.get(futures)) # [0, 1, 4, 9]

@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

counters = [Counter.remote() for i in range(4)]
[c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]
print(ray.get(futures)) # [1, 1, 1, 1]

# Note the following behaviors:
# The second task will not be executed until the first task has finished executing because the second task depends on the output of the first task.
# If the two tasks are scheduled on different machines, the output of the first task (the value corresponding to obj_ref1/objRef1) will be sent over the network to the machine where the second task is scheduled.

# MWE for tiny work parallelization
def tiny_work(x):
    time.sleep(0.0001) # replace this is with work you need to do
    return x

@ray.remote
def mega_work(start, end):
    return [tiny_work(x) for x in range(start, end)]

start = time.time()
result_ids = []
[result_ids.append(mega_work.remote(x*1000, (x+1)*1000)) for x in range(100)]
results = ray.get(result_ids)
print("duration =", time.time() - start)

# Close down at the end of the session
ray.shutdown()