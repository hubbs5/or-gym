# or-gym
## Environments for OR and RL Research

This library contains environments consisting of operations research problems which adhere to the OpenAI Gym API. The purpose is to bring reinforcement learning to the operations research community via accessible simulation environments featuring classic problems that are solved both with reinforcement learning as well as traditional OR techniques.

## Requirements

<font color="#ff2400">See `requirements.txt`. Add to this...</font>

## Installation

Clone the repo to your machine. Then navigate to the `or_gym_envs` folder. From there, run `pip install -e .` to install the OR Gym environments. To test the installation, run the `or_gym_test.py` file. You should receive a printout saying that the model was loaded successfully.
<font color="#ff2400">Eventually, we need to get a simple `pip install or-gym` command.</font>

## Environments

- `Knapsack-v0`: a small version of the classic unbounded knapsack problem (<font color="#ff2400">find a simple one in the lit to cite</font>).
- `Knapsack-v1`: bounded knapsack problem with uncertainty (<font color="#ff2400">to be implemented</font>).
- `NewsVendor-v0`: <font color="#ff2400">to be implemented</font>
- `TSP-v0`: <font color="#ff2400">to be implemented</font>
- `VRP-v0`: <font color="#ff2400">to be implemented</font>
- `RTN-v0`: <font color="#ff2400">to be implemented</font>

Others to implement such as network design under uncertainty, portfolio optimization, etc? ChemE specific models like max pooling? 

Two key requirements for RL to be effective need to be kept in mind while finding models in the literature:
- Requires sequential decision making such that the decisions made at each time period depend on decisions made in previous time periods.
- Requires uncertainty in the model.
These two factors tend to be difficult for many existing approaches in to deal with, but are where RL is particularly effective in comparison.