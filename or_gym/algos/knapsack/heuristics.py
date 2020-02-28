import numpy as np

def ukp_heuristic(env):
    assert env.spec.id == 'Knapsack-v0', \
        '{} received. Heuristic designed for Knapsack-v0.'.format(env.spec.id)
    env.reset()
    
    # Get value-weight ratios
    vw_ratio = env.item_values / env.item_weights
    vw_order = env.item_numbers[np.argsort(vw_ratio)[::-1]]
    actions = []
    rewards = []
    done = False
    while not done:
        max_item = vw_order[0]
        # Check that item fits
        if env.item_weights[max_item] > (env.max_weight - env.current_weight):
            # Remove item from list
            vw_order = vw_order[1:].copy()
            continue
        # Select max_item
        state, reward, done, _ = env.step(max_item)
        actions.append(max_item)
        rewards.append(reward)
        
    return actions, rewards

def bkp_heuristic(env):
    assert env.spec.id == 'Knapsack-v1', \
        '{} received. Heuristic designed for Knapsack-v1.'.format(env.spec.id)
    env.reset()

    # Get value-weight ratios
    vw_ratio = env.item_values / env.item_weights
    vw_order = env.item_numbers[np.argsort(vw_ratio)[::-1]]
    actions = []
    rewards = []
    done = False
    while not done:
        # Check that max item is available
        max_item = vw_order[0]
        if env.item_limits[max_item] == 0:
            # Remove item from list
            vw_order = vw_order[1:].copy()
            continue
        # Check that item fits
        if env.item_weights[max_item] > (env.max_weight - env.current_weight):
            # Remove item from list
            vw_order = vw_order[1:].copy()
            continue
        # Select max_item
        state, reward, done, _ = env.step(max_item)
        actions.append(max_item)
        rewards.append(reward)
        
    return actions, rewards

def okp_heuristic(env):
    assert env.spec.id == 'Knapsack-v2', \
        '{} received. Heuristic designed for Knapsack-v2.'.format(env.spec.id)
    env.reset()

    vw_ratio = env.item_values / env.item_weights
    T = np.mean(vw_ratio)
    item = copy.copy(env.current_item)
    done = False
    actions = []
    items_taken = []
    items_offered = []
    rewards = []
    count = 0
    while not done:
        if env.item_weights[item] >= (env.max_weight - env.current_weight):
            action = 0
        elif vw_ratio[item] >= T / (1 + count):
            action = 1
            items_taken.append(item)
        else:
            action = 0
        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        items_offered.append(item)
        item = state[-1][-1]
        count += 1

    return actions, items_offered, rewards