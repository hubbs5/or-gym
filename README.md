# or-gym
## Environments for OR and RL Research

This library contains environments consisting of operations research problems which adhere to the OpenAI Gym API. The purpose is to bring reinforcement learning to the operations research community via accessible simulation environments featuring classic problems that are solved both with reinforcement learning as well as traditional OR techniques.

## Installation

This library requires Python 3.5+ in order to function. 
Installation is possible via `pip`:

`$ pip install or-gym`

Or, you can install directly from GitHub with:

```
git clone https://github.com/hubbs5/or-gym.git
cd or-gym
pip install -e .
```

## Citation

## Environments

- `Knapsack-v0`: a small version of the classic unbounded knapsack problem with 200 items.
- `Knapsack-v1`: binary (0-1) knapsack problem with 200 items.
- `Knapsack-v2`: bounded knapsack problem with 200 items.
- `Knapsack-v3`: stochastic, online knapsack problem.
- `BinPacking-v0` through `BinPacking-v5`: online bin packing problem taken from [Balaji et al.](https://arxiv.org/abs/1911.10641).
- `Newsvendor-v0`: multi-period newsvendor problem with lead times taken from [Balaji et al.](https://arxiv.org/abs/1911.10641).
- `VMPacking-v0`: permanent, multi-dimensional virtual machine packing problem.
- `VMPacking-v1`: temporary, multi-dimensional virtual machine packing problem.
- `VehicleRouting-v0`: pick-up and delivery problem with delivery windows taken from [Balaji et al.](https://arxiv.org/abs/1911.10641).
- `InvManagement-v0`: multi-echelon supply chain re-order problem with backlogs.
- `InvManagement-v1`: multi-echelon supply chain re-order problem without backlog.
- `MPAA-v0`: Multi-period asset allocation problem for managing investment decisions taken from [Dantzig and Infanger](https://apps.dtic.mil/dtic/tr/fulltext/u2/a242510.pdf).
- `TSP-v0`: travelling salesman problem with bi-directional connections and uniform cost.
- `TSP-v1`: travelling salesman problem with bi-directional connections.
- `TSP-v2`: travelling salesman problem for 50 largest US cities.
- `TSP-v3`: travelling salesman problem with stochastic costs and connections.

## Resources
