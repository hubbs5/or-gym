import numpy as np

def first_fit_heuristic(env):
    # assert env.spec.id == ('VMPacking-v0' or 'VMPacking-v1'), \
        # '{} received. Heuristic designed for VMPacking-v0/v1.'.format(env.spec.id)

    state = env.reset()
    done = False
    rewards, actions = [], []
    while done == False:
        if type(state) == dict:
            state = state['state'].copy()
        action = first_fit_step(state)
        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)

    return actions, rewards

def next_fit_heuristic(env):
    assert env.spec.id == ('VMPacking-v0' or 'VMPacking-v1'), \
        '{} received. Heuristic designed for VMPacking-v0/v1.'.format(env.spec.id)

    state = env.reset()
    done = False
    rewards, actions = [], []
    while done == False:
        if type(state) == dict:
            state = state['state'].copy()
        action = next_fit_step(state)
        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)

    return actions, rewards

# First fit: Pack item into lowest current bin where it fits, else into a new bin
def first_fit_step(state):
    s_bins, s_item = state[:-1], state[-1, 1:]
    action = None
    open_bins = np.where(s_bins[:,0]==1)[0]
    if len(open_bins) < 1:
        # Open first bin for item
        action = 0
    else:
        # Check each bin until one is found to fit the item
        for b in open_bins:
            if all(s_bins[b, [1, 2]] + s_item <= 1):
                action = b
        if action is None:
            action = np.max(open_bins) + 1
    return action

# Next fit: Pack item into current bin, else into a new bin
def next_fit_step(state):
    s_bins, s_item = state[:-1], state[-1, 1:]
    action = None
    current_bin = np.where(s_bins[:,0]==1)[0]
    if len(current_bin) < 1:
        # Open first bin
        action = 0
    else:
        # Check if it fits into current bin
        b = current_bin[-1]
        if all(s_bins[b, [1, 2]] + s_item <= 1):
            action = b
        else:
            action = b + 1
    return action    