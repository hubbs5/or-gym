import numpy as np
from collections import Iterable
import copy

def ukp_heuristic(env):
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
        if env.item_weights[max_item] + env.current_weight > env.max_weight:
            # Remove item from list
            vw_order = vw_order[1:].copy()
            if len(vw_order) == 0:
                # End episode
                break
            else:
                continue
        # Select max_item
        state, reward, done, _ = env.step(max_item)
        actions.append(max_item)
        rewards.append(reward)
        
    return actions, rewards

def bkp_heuristic(env):
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
        cont_flag = False
        if env.item_limits[max_item] == 0:
            # Remove item from list
            vw_order = vw_order[1:].copy()
            cont_flag = True
        # Check that item fits
        elif env.item_weights[max_item] + env.current_weight > env.max_weight:
            # Remove item from list
            vw_order = vw_order[1:].copy()
            cont_flag = True
        if len(vw_order) == 0:
            # End episode
            break
        if cont_flag:
            continue
        # Select max_item
        state, reward, done, _ = env.step(max_item)
        actions.append(max_item)
        rewards.append(reward)
        
    return actions, rewards

def okp_heuristic(env, scenario=None):
    '''TwoBins from Han 2015'''
    if scenario is not None:
        # Ensure scenario is iterable of length step_limit
        assert isinstance(scenario, Iterable), 'scenario not iterable.'
        assert len(scenario) >= env.step_limit, 'scenario too short.'
    env.reset()
    
    done = False
    actions = []
    items_offered = []
    rewards = []
    r = bool(np.random.choice([0, 1]))
    count, rejection_weight = 0, 0
    while not done:
        if scenario is not None:
            item = scenario[count]
        else:
            item = copy.copy(env.current_item)
        action = 0
        if r:
            # Greedy algorithm
            if env.item_weights[max_item] + env.current_weight > env.max_weight:
                action = 1
        else:
        	rejection_weight += env.item_weights[item]
        	if rejection_weight > env.max_weight:
        		action = 1

        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        items_offered.append(item)
        count += 1

    return actions, items_offered, rewards