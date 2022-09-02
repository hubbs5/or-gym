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
## Quickstart Example and Benchmarking Example 

See the IPython notebook entitled `inv-management-quickstart.ipynb` in the `examples` folder for a quickstart example for training an agent in an OR-GYM environemnt, and for using the environment for benchmarking policies found by other algorithms. For the RL algorithm, Ray 1.0.0 is required.

## Citation
```
@misc{HubbsOR-Gym,
    author={Christian D. Hubbs and Hector D. Perez and Owais Sarwar and Nikolaos V. Sahinidis and Ignacio E. Grossmann and John M. Wassick},
    title={OR-Gym: A Reinforcement Learning Library for Operations Research Problems},
    year={2020},
    Eprint={arXiv:2008.06319}
}
```

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
- `NetworkManagement-v0`: multi-echelon supply chain network problem with backlogs from [Perez et al.](https://www.mdpi.com/2227-9717/9/1/102).
- `NetworkManagement-v1`: multi-echelon supply chain network problem without backlogs from [Perez et al.](https://www.mdpi.com/2227-9717/9/1/102).
- `PortfolioOpt-v0`: Multi-period asset allocation problem for managing investment decisions taken from [Dantzig and Infanger](https://apps.dtic.mil/dtic/tr/fulltext/u2/a242510.pdf).
- `TSP-v0`: traveling salesman problem with bi-directional connections and uniform cost.
- `TSP-v1`: traveling salesman problem with bi-directional connections.

## Resources

Information on results and supporting models can be found [here](https://arxiv.org/abs/2008.06319).

## Examples

- [Action Masking with RLlib using the Knapsack Environment](https://www.datahubbs.com/action-masking-with-rllib/)
- [How to Use Deep Reinforcement Learning to Improve your Supply Chain](https://www.datahubbs.com/how-to-use-deep-reinforcement-learning-to-improve-your-supply-chain/)