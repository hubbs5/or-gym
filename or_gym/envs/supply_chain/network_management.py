'''
Multi-period inventory management
Hector Perez, Christian Hubbs, Can Li
9/14/2020
'''

import gym
import itertools
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import *
from or_gym.utils import assign_env_config
from collections import deque
import matplotlib.pyplot as plt

class NetInvMgmtMasterEnv(gym.Env):
    '''
    The supply network environment is structured as follows:
    
    It is a multi-period multi-node production-inventory system for 
    a single non-perishable product that is sold in discrete quantities. 
    Two main types of nodes exist: 1) production nodes (which have an 
    inventory holding area and a manufacturing area), and 2) distribution
    nodes (which only have an inventory holding area). Retail nodes are
    considered distribution nodes. Other node types in the network include 
    1) raw material nodes (source nodes), which have unlimited supply
    of raw materials, and 2) market nodes (sink nodes), which generate an 
    uncertain demand on their respective retailers in each period. 

    Within production nodes, the inventory holding area holds the inventory 
    necessary to produce the respective intermediate material at that node. 
    Yield ratios are specified at each production stage relating the amount 
    of material produced from one unit of inventory. Production at each node
    is bounded by the nodes's production capacity and the available inventory.

    Lead times between neighbor nodes exist and are associated with the edges
    connecting them.
        
    At the beginning of each time period, the following sequence of events occurs:
    
    0) Each node places replenishment orders to their immediate suppliers. 
       Replenishment orders are filled according to available production capacity 
       and available inventory at the suppliers. There is a cost associated with
       each order request.
    1) Incoming inventory replenishment shipments that have made it down network
       pipeline (after the associated lead time) are received at each node.
    2) Market demands occur at the retail nodes. Demands are sampled from a 
       specified discrete probability distribution. Demands are filled according 
       to available inventory at the retailers.
    4) Option: one of the following occurs,
        a) Unfulfilled sales are backlogged at a penalty. 
            Note: Backlogged orders are added to the next period's market demand.
        b) Unfulfilled sales and replenishment orders are lost with a 
           goodwill-loss penalty. 
    5) Surpluss inventory is held at each stage at a holding cost. 
        Pipeline inventories also incur a cost for each period in the pipeline.
        
    '''
    def __init__(self, *args, **kwargs):
        '''
        num_periods = number of periods in simulation.
        Node specific parameters:
            - I0 = initial inventory.
            - C = production capacity.
            - v = production yield in the range (0, 1].
            - o = unit operating cost (feed-based)
            - h = unit holding cost for excess on-hand inventory.
        Edge specific parameters:
            - L = lead times in betwen adjacent nodes.
            - p = unit price to send material between adjacent nodes (purchase price/reorder cost)
            - b = unit backlog cost or good-wil loss for unfulfilled market demand between adjacent retailer and market.
            - g = unit holding cost for pipeline inventory on a specified edge.
            - prob_dist = probability distribution function on a (retailer, market) edge.
            - demand_dist = demand distribution for (retailer, market) edge. Two options:
                - use scipy probability distribution: must be a lambda function calling the rvs method of the distribution
                    i.e. lambda: poisson.rvs(mu=20)
                - use a list of user specified demands for each period. 
        backlog = Are unfulfilled orders backlogged? True = backlogged, False = lost sales.
        demand_dist = distribution function for customer demand (e.g. poisson, binomial, uniform, geometric, etc.)
        dist_param = named values for parameters fed to statistical distribution.
            poisson: {'mu': <mean value>}
            binom: {'n': <mean value>, 
                    'p': <probability between 0 and 1 of getting the mean value>}
            raindint: {'low' = <lower bound>, 'high': <upper bound>}
            geom: {'p': <probability. Outcome is the number of trials to success>}
        alpha = discount factor in the range (0,1] that accounts for the time value of money
        seed_int = integer seed for random state.
        user_D = dictionary containing user specified demand (list) for each (retail, market) pair at
            each time period in the simulation. If all zeros, ignored; otherwise, demands will be taken from this list.
        sample_path = dictionary specifying if is user_D (for each (retail, market) pair) is sampled from demand_dist.
        '''
        # set default (arbitrary) values when creating environment (if no args or kwargs are given)
        self._max_rewards = 2000
        self.num_periods = 30
        self.backlog = True
        self.alpha = 1.00
        self.seed_int = 0
        self.user_D = {(1,0): np.zeros(self.num_periods)}
        self.sample_path = {(1,0): False}
        self._max_rewards = 2000

        # create graph
        self.graph = nx.DiGraph()
        # Market 
        self.graph.add_nodes_from([0])
        # Retailer
        self.graph.add_nodes_from([1], I0 = 100,
                                        h = 0.030)
        # Distributors
        self.graph.add_nodes_from([2], I0 = 110,
                                        h = 0.020)
        self.graph.add_nodes_from([3], I0 = 80,
                                        h = 0.015)
        # Manufacturers
        self.graph.add_nodes_from([4], I0 = 400,
                                        C = 90,
                                        o = 0.010,
                                        v = 1.000,
                                        h = 0.012)
        self.graph.add_nodes_from([5], I0 = 350,
                                        C = 90,
                                        o = 0.015,
                                        v = 1.000,
                                        h = 0.013)
        self.graph.add_nodes_from([6], I0 = 380,
                                        C = 80,
                                        o = 0.012,
                                        v = 1.000,
                                        h = 0.011)
        # Raw materials
        self.graph.add_nodes_from([7, 8])
        # Links
        self.graph.add_edges_from([(1,0,{'p': 2.000,
                                         'b': 0.100,
                                         'demand_dist': poisson,
                                         'dist_param': {'mu': 20}}),
                                   (2,1,{'L': 5,
                                         'p': 1.500,
                                         'g': 0.010}),
                                   (3,1,{'L': 3,
                                         'p': 1.600,
                                         'g': 0.015}),
                                   (4,2,{'L': 8,
                                         'p': 1.000,
                                         'g': 0.008}),
                                   (4,3,{'L': 10,
                                         'p': 0.800,
                                         'g': 0.006}),
                                   (5,2,{'L': 9,
                                         'p': 0.700,
                                         'g': 0.005}),
                                   (6,2,{'L': 11,
                                         'p': 0.750,
                                         'g': 0.007}),
                                   (6,3,{'L': 12,
                                         'p': 0.800,
                                         'g': 0.004}),
                                   (7,4,{'L': 0,
                                         'p': 0.150,
                                         'g': 0.000}),
                                   (7,5,{'L': 1,
                                         'p': 0.050,
                                         'g': 0.005}),
                                   (8,5,{'L': 2,
                                         'p': 0.070,
                                         'g': 0.002}),
                                   (8,6,{'L': 0,
                                         'p': 0.200,
                                         'g': 0.000})])
        
        # add environment configuration dictionary and keyword arguments
        assign_env_config(self, kwargs)

        # Save user_D and sample_path to graph metadata
        for link in self.user_D.keys():
            d = self.user_D[link]
            if np.sum(d) != 0:
                self.graph.edges[link]['user_D'] = d
                if link in self.sample_path.keys():
                    self.graph.edges[link]['sample_path'] = self.sample_path[link]
            else:
                # Placeholder to avoid key errors
                self.graph.edges[link]['user_D'] = 0
        
        self.num_nodes = self.graph.number_of_nodes()
        self.adjacency_matrix = np.vstack(self.graph.edges())
        # Set node levels
        self.levels = {}
        self.levels['retailer'] = np.array([1])
        self.levels['distributor'] = np.unique(np.hstack(
            [list(self.graph.predecessors(i)) for i in self.levels['retailer']]))
        self.levels['manufacturer'] = np.unique(np.hstack(
            [list(self.graph.predecessors(i)) for i in self.levels['distributor']]))
        self.levels['raw_materials'] = np.unique(np.hstack(
            [list(self.graph.predecessors(i)) for i in self.levels['manufacturer']]))

        self.level_col = {'retailer': 0,
                    'distributor': 1,
                    'manufacturer': 2,
                    'raw_materials': 3}

        self.market = [j for j in self.graph.nodes() if len(list(self.graph.successors(j))) == 0]
        self.distrib = [j for j in self.graph.nodes() if 'C' not in self.graph.nodes[j] and 'I0' in self.graph.nodes[j]]
        self.retail = [j for j in self.graph.nodes() if len(set.intersection(set(self.graph.successors(j)), set(self.market))) > 0]
        self.factory = [j for j in self.graph.nodes() if 'C' in self.graph.nodes[j]]
        self.rawmat = [j for j in self.graph.nodes() if len(list(self.graph.predecessors(j))) == 0]
        self.main_nodes = np.sort(self.distrib + self.factory)
        self.reorder_links = [e for e in self.graph.edges() if 'L' in self.graph.edges[e]] #exclude links to markets (these cannot have lead time 'L')
        self.retail_links = [e for e in self.graph.edges() if 'L' not in self.graph.edges[e]] #links joining retailers to markets
        self.network_links = [e for e in self.graph.edges()] #all links involved in sale in the network

        # check inputs
        assert set(self.graph.nodes()) == set.union(set(self.market),
                                                    set(self.distrib),
                                                    set(self.factory),
                                                    set(self.rawmat)), "The union of market, distribution, factory, and raw material nodes is not equal to the system nodes."
        for j in self.graph.nodes():
            if 'I0' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['I0'] >= 0, "The initial inventory cannot be negative for node {}.".format(j)
            if 'h' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['h'] >= 0, "The inventory holding costs cannot be negative for node {}.".format(j)
            if 'C' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['C'] > 0, "The production capacity must be positive for node {}.".format(j)
            if 'o' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['o'] >= 0, "The operating costs cannot be negative for node {}.".format(j)
            if 'v' in self.graph.nodes[j]:
                assert self.graph.nodes[j]['v'] > 0 and self.graph.nodes[j]['v'] <= 1, "The production yield must be in the range (0, 1] for node {}.".format(j)
        for e in self.graph.edges():
            if 'L' in self.graph.edges[e]:
                assert self.graph.edges[e]['L'] >= 0, "The lead time joining nodes {} cannot be negative.".format(e)
            if 'p' in self.graph.edges[e]:
                assert self.graph.edges[e]['p'] >= 0, "The sales price joining nodes {} cannot be negative.".format(e)
            if 'b' in self.graph.edges[e]:
                assert self.graph.edges[e]['b'] >= 0, "The unfulfilled demand costs joining nodes {} cannot be negative.".format(e)
            if 'g' in self.graph.edges[e]:
                assert self.graph.edges[e]['g'] >= 0, "The pipeline inventory holding costs joining nodes {} cannot be negative.".format(e)
            if 'sample_path' in self.graph.edges[e]:
                assert isinstance(self.graph.edges[e]['sample_path'], bool), "When specifying if a user specified demand joining (retailer, market): {} is sampled from a distribution, sample_path must be a Boolean.".format(e)
            if 'demand_dist' in self.graph.edges[e]:
                dist = self.graph.edges[e]['demand_dist'] #extract distribution
                assert dist.cdf(0,**self.graph.edges[e]['dist_param']), "Wrong parameters passed to the demand distribution joining (retailer, market): {}.".format(e)
        assert self.backlog == False or self.backlog == True, "The backlog parameter must be a boolean."
        assert self.graph.number_of_nodes() >= 2, "The minimum number of nodes is 2. Please try again"
        assert self.alpha>0 and self.alpha<=1, "alpha must be in the range (0, 1]."
        
        # set random generation seed (unless using user demands)
        self.seed(self.seed_int)
        
        # action space (reorder quantities for each node for each supplier; list)
        # An action is defined for every node
        num_reorder_links = len(self.reorder_links) 
        self.lt_max = np.max([self.graph.edges[e]['L'] for e in self.graph.edges() if 'L' in self.graph.edges[e]])
        self.init_inv_max = np.max([self.graph.nodes[j]['I0'] for j in self.graph.nodes() if 'I0' in self.graph.nodes[j]])
        self.capacity_max = np.max([self.graph.nodes[j]['C'] for j in self.graph.nodes() if 'C' in self.graph.nodes[j]])
        self.pipeline_length = sum([self.graph.edges[e]['L']
            for e in self.graph.edges() if 'L' in self.graph.edges[e]])
        self.lead_times = {e: self.graph.edges[e]['L'] 
            for e in self.graph.edges() if 'L' in self.graph.edges[e]}
        self.obs_dim = self.pipeline_length + len(self.main_nodes) + len(self.retail_links)
        # self.pipeline_length = len(self.main_nodes)*(self.lt_max+1)
        self.action_space = gym.spaces.Box(
            low=np.zeros(num_reorder_links),
            high=np.ones(num_reorder_links)*(self.init_inv_max + self.capacity_max*self.num_periods), 
            dtype=np.int32)
        # observation space (total inventory at each node, which is any integer value)
        self.observation_space = gym.spaces.Box(
            low=np.ones(self.obs_dim)*np.iinfo(np.int32).min,
            high=np.ones(self.obs_dim)*np.iinfo(np.int32).max,
            dtype=np.int32)
            # low=-np.ones(self.pipeline_length)*(self.init_inv_max + self.capacity_max*self.num_periods)*10,
            # high=np.ones(self.pipeline_length)*(self.init_inv_max + self.capacity_max*self.num_periods), 
            # dtype=np.int32)

        # intialize
        self.reset()

    def seed(self,seed=None):
        '''
        Set random number generation seed
        '''
        # seed random state
        if seed != None:
            np.random.seed(seed=int(seed))
        
    def _RESET(self):
        '''
        Create and initialize all variables and containers.
        Nomenclature:
            I = On hand inventory at the start of each period at each stage (except last one).
            T = Pipeline inventory at the start of each period at each stage (except last one).
            R = Replenishment order placed at each period at each stage (except last one).
            D = Customer demand at each period (at the retailer)
            S = Sales performed at each period at each stage.
            B = Backlog at each period at each stage.
            LS = Lost sales at each period at each stage.
            P = Total profit at each stage.
        '''
        T = self.num_periods
        J = len(self.main_nodes)
        RM = len(self.retail_links)  # number of retailer-market pairs
        PS = len(self.reorder_links) # number of purchaser-supplier pairs in the network
        SL = len(self.network_links) # number of edges in the network (excluding links form raw material nodes)
        
        # simulation result lists
        self.X=pd.DataFrame(data = np.zeros([T + 1, J]), 
                            columns = self.main_nodes) # inventory at the beginning of each period
        self.Y=pd.DataFrame(data = np.zeros([T + 1, PS]), 
                            columns = pd.MultiIndex.from_tuples(self.reorder_links,
                            names = ['Source','Receiver'])) # pipeline inventory at the beginning of each period
        self.R=pd.DataFrame(data = np.zeros([T, PS]), 
                            columns = pd.MultiIndex.from_tuples(self.reorder_links,
                            names = ['Supplier','Requester'])) # replenishment orders
        self.S=pd.DataFrame(data = np.zeros([T, SL]), 
                            columns = pd.MultiIndex.from_tuples(self.network_links, 
                            names = ['Seller','Purchaser'])) # units sold
        self.D=pd.DataFrame(data = np.zeros([T, RM]), 
                            columns = pd.MultiIndex.from_tuples(self.retail_links, 
                            names = ['Retailer','Market'])) # demand at retailers
        self.U=pd.DataFrame(data = np.zeros([T, RM]), 
                            columns = pd.MultiIndex.from_tuples(self.retail_links, 
                            names = ['Retailer','Market'])) # unfulfilled demand for each market - retailer pair
        self.P=pd.DataFrame(data = np.zeros([T, J]), 
                            columns = self.main_nodes) # profit at each node
        
        # initializetion
        self.period = 0 # initialize time
        for j in self.main_nodes:
            self.X.loc[0,j]=self.graph.nodes[j]['I0'] # initial inventory
        self.Y.loc[0,:]=np.zeros(PS) # initial pipeline inventory
        self.action_log = np.zeros([T, PS])

        # set state
        self._update_state()
        
        return self.state

    def _update_state(self):
        # State is a concatenation of demand, inventory, and pipeline at each time step
        demand = np.hstack([self.D[d].iloc[self.period] for d in self.retail_links])
        inventory = np.hstack([self.X[n].iloc[self.period] for n in self.main_nodes])

        # Pipeline values won't be of proper dimension if current
        # current period < lead time. We need to add 0's as padding.
        if self.period == 0:
            _pipeline = [[self.Y[k].iloc[0]]
                for k, v in self.lead_times.items()]
        else:
            _pipeline = [self.Y[k].iloc[max(self.period-v,0):self.period].values
                for k, v in self.lead_times.items()]
        pipeline = []
        for p, v in zip(_pipeline, self.lead_times.values()):
            if v == 0:
                continue
            if len(p) <= v:
                pipe = np.zeros(v)
                pipe[-len(p):] += p
            pipeline.append(pipe)
        pipeline = np.hstack(pipeline)
        self.state = np.hstack([demand, inventory, pipeline])
    
    def _STEP(self, action):
        '''
        Take a step in time in the multiperiod inventory management problem.
        action = number of units to request from each supplier.
            dictionary: keys are (supplier, purchaser) tuples
                        values are number of units requested from supplier
                        dimension = len(reorder_links) (number of edges joining all nodes, 
                                                        except market nodes)
        '''
        t = self.period
        if type(action) != dict: # convert to dict if a list was given
            action = {key: action[i] for i, key in enumerate(self.reorder_links)}
        
        # Place Orders
        for key in action.keys():
            request = round(max(action[key],0)) # force to integer value
            supplier = key[0]
            purchaser = key[1]
            if supplier in self.rawmat:
                self.R.loc[t,(supplier, purchaser)] = request # accept request since supply is unlimited
                self.S.loc[t,(supplier, purchaser)] = request
            elif supplier in self.distrib:
                X_supplier = self.X.loc[t,supplier] # request limited by available inventory at beginning of period
                self.R.loc[t,(supplier, purchaser)] = min(request, X_supplier)
                self.S.loc[t,(supplier, purchaser)] = min(request, X_supplier)
            elif supplier in self.factory:
                C = self.graph.nodes[supplier]['C'] # supplier capacity
                v = self.graph.nodes[supplier]['v'] # supplier yield
                X_supplier = self.X.loc[t,supplier] # on-hand inventory at beginning of period
                self.R.loc[t,(supplier, purchaser)] = min(request, C, v*X_supplier)
                self.S.loc[t,(supplier, purchaser)] = min(request, C, v*X_supplier)
            
        #Receive deliveries and update inventories
        for j in self.main_nodes:
            #update pipeline inventories
            incoming = []
            for k in self.graph.predecessors(j):
                L = self.graph.edges[(k,j)]['L'] #extract lead time
                if t - L >= 0: #check if delivery has arrived
                    delivery = self.R.loc[t-L,(k,j)]
                else:
                    delivery = 0
                incoming += [delivery] #update incoming material
                self.Y.loc[t+1,(k,j)] = self.Y.loc[t,(k,j)] - delivery + self.R.loc[t,(k,j)]

            #update on-hand inventory
            if 'v' in self.graph.nodes[j]: #extract production yield
                v = self.graph.nodes[j]['v']
            else:
                v = 1
            outgoing = 1/v * np.sum([self.S.loc[t,(j,k)] for k in self.graph.successors(j)]) #consumed inventory (for requests placed)
            self.X.loc[t+1,j] = self.X.loc[t,j] + np.sum(incoming) - outgoing
            
        # demand is realized
        for j in self.retail:
            for k in self.market:
                #read user specified demand. if all zeros, use demand_dist instead.
                Demand = self.graph.edges[(j,k)]['user_D']
                if np.sum(Demand) > 0:
                    self.D.loc[t,(j,k)] = Demand[t]
                else:
                    Demand = self.graph.edges[(j,k)]['demand_dist']
                    self.D.loc[t,(j,k)] = Demand.rvs(
                        **self.graph.edges[(j,k)]['dist_param'])
                if self.backlog and t >= 1:
                    D = self.D.loc[t,(j,k)] + self.U.loc[t-1,(j,k)]
                else:
                    D = self.D.loc[t,(j,k)]
                #satisfy demand up to available level
                X_retail = self.X.loc[t+1,j] #get inventory at retail before demand was realized
                self.S.loc[t,(j,k)] = min(D, X_retail) #perform sale
                self.X.loc[t+1,j] -= self.S.loc[t,(j,k)] #update inventory
                self.U.loc[t,(j,k)] = D - self.S.loc[t,(j,k)] #update unfulfilled orders

        # calculate profit
        for j in self.main_nodes:
            a = self.alpha
            SR = np.sum([self.graph.edges[(j,k)]['p'] * self.S.loc[t,(j,k)] for k in self.graph.successors(j)]) #sales revenue
            PC = np.sum([self.graph.edges[(k,j)]['p'] * self.R.loc[t,(k,j)] for k in self.graph.predecessors(j)]) #purchasing costs
            if j not in self.rawmat:
                HC = self.graph.nodes[j]['h'] * self.X.loc[t+1,j] + np.sum([self.graph.edges[(k,j)]['g'] * self.Y.loc[t+1,(k,j)] for k in self.graph.predecessors(j)]) #holding costs
            else:
                HC = 0
            if j in self.factory:
                OC = self.graph.nodes[j]['o'] / self.graph.nodes[j]['v'] * np.sum([self.S.loc[t,(j,k)] for k in self.graph.successors(j)]) #operating costs
            else:
                OC = 0
            if j in self.retail:
                UP = np.sum([self.graph.edges[(j,k)]['b'] * self.U.loc[t,(j,k)] for k in self.graph.successors(j)]) #unfulfilled penalty
            else:
                UP = 0
            self.P.loc[t,j] = a**t * (SR - PC - OC - HC - UP)
        
        # update period
        self.period += 1

        # set reward (profit from current timestep)
        reward = self.P.loc[t,:].sum()
        
        # determine if simulation should terminate
        if self.period >= self.num_periods:
            done = True
        else:
            done = False
            # update stae
            self._update_state()

        return self.state, reward, done, {}
    
    def sample_action(self):
        '''
        Generate an action by sampling from the action_space
        '''
        return self.action_space.sample()

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()

    def plot_network(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        adjacency_matrix = np.vstack(self.graph.edges())
        # Set level colors
        level_col = {'retailer': 0,
                    'distributor': 1,
                    'manufacturer': 2,
                    'raw_materials': 3}

        max_density = np.max([len(v) for v in self.levels.values()])
        node_coords = {}
        node_num = 1
        plt.figure(figsize=(12,8))
        for i, (level, nodes) in enumerate(self.levels.items()):
            n = len(nodes)
            node_y = max_density / 2 if n == 1 else np.linspace(0, max_density, n)
            node_y = np.atleast_1d(node_y)
            plt.scatter(np.repeat(i, n), node_y, label=level, s=50)
            for y in node_y:
                plt.annotate(r'$N_{}$'.format(node_num), xy=(i, y+0.05))
                node_coords[node_num] = (i, y)
                node_num += 1

        # Draw edges
        for node_num, v in node_coords.items():
            x, y = v
            sinks = adjacency_matrix[np.where(adjacency_matrix[:, 0]==node_num)][:, 1]
            for s in sinks:
                try:
                    sink_coord = node_coords[s]
                except KeyError:
                    continue
                for k, n in self.levels.items():
                    if node_num in n:
                        color = colors[level_col[k]]
                x_ = np.hstack([x, sink_coord[0]])
                y_ = np.hstack([y, sink_coord[1]])
                plt.plot(x_, y_, color=color)

        plt.ylabel('Node')
        plt.yticks([0], [''])
        plt.xlabel('Level')
        plt.xticks(np.arange(len(self.levels)), [k for k in self.levels.keys()])
        plt.show()
        
class NetInvMgmtBacklogEnv(NetInvMgmtMasterEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class NetInvMgmtLostSalesEnv(NetInvMgmtMasterEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backlog = False