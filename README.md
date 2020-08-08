# or-gym
## Environments for OR and RL Research

This library contains environments consisting of operations research problems which adhere to the OpenAI Gym API. The purpose is to bring reinforcement learning to the operations research community via accessible simulation environments featuring classic problems that are solved both with reinforcement learning as well as traditional OR techniques.

## Installation

OR-Gym requires Python 3.5+, Numpy, Scipy, and Gym.

Installation is possible via `pip`:

`$ pip install or-gym`

## Environments

- `Knapsack-v0`: a small version of the classic unbounded knapsack problem with 200 items.
- `Knapsack-v1`: binary (0-1) knapsack problem with 200 items.
- `Knapsack-v2`: bounded knapsack problem with 200 items.
- `Knapsack-v3`: stochastic, online knapsack problem.
- `BinPacking-v0`: 


Others to implement such as network design under uncertainty, portfolio optimization, etc? ChemE specific models like max pooling? 

Two key requirements for RL to be effective need to be kept in mind while finding models in the literature:
- Requires sequential decision making such that the decisions made at each time period depend on decisions made in previous time periods.
- Requires uncertainty in the model.
These two factors tend to be difficult for many existing approaches in to deal with, but are where RL is particularly effective in comparison.