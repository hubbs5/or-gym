import numpy as np
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils import assign_env_config
from copy import copy, deepcopy

class TSPEnv(gym.Env):
    '''
    Bi-directional connections and uniform cost


    '''
    def __init__(self, *args, **kwargs):
        self.N = 10
        self.nodes = np.arange(self.N)
        self.move_cost = -1
        self.invalid_action_cost = -1000
        self.step_limit = 2*self.N
        self.mask = False
        assign_env_config(self, kwargs)
        self.obs_dim = 1+self.N**2
        obs_space = spaces.Box(-1, self.N, shape=(self.obs_dim,), dtype=np.int32)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.N,), dtype=np.int8),
                "avail_actions": spaces.Box(0, 1, shape=(self.N,), dtype=np.int8),
                "state": obs_space
            })
        else:
            self.observation_space = obs_space
        self.action_space = spaces.Discrete(self.N)
        
        self.reset()
        
    def step(self, action):
        done = False
        connections = self.node_dict[self.current_node]
        # Invalid action
        if action not in connections:
            reward = self.invalid_action_cost
        # Move to new node
        else:
            self.current_node = action
            reward = self.move_cost
            self.visit_log[self.current_node] += 1
            
        self.state = self._update_state()
        self.step_count += 1
        # See if all nodes have been visited
        unique_visits = sum([1 if v > 0 else 0 
            for v in self.visit_log.values()])
        if unique_visits >= self.N:
            done = True
            reward += 1000
        if self.step_count >= self.step_limit:
            done = True
            
        return self.state, reward, done, {}
        
    def reset(self):
        self.step_count = 0
        self._generate_connections()
        self.current_node = np.random.choice(self.nodes)
        self.visit_log = {n: 0 for n in self.nodes}
        self.visit_log[self.current_node] += 1
        
        self.state = self._update_state()
        return self.state
        
    def _update_state(self):
        node_connections = self.connection_matrix.copy()
        # Set value to 1 for existing, un-visited nodes
        # Set value to -1 for existing, visited nodes
        # Set value to 0 if connection doesn't exist
        visited = np.array([bool(min(v, 1))
            for v in self.visit_log.values()])
        node_connections[:, visited] = -1
        node_connections[np.where(self.connection_matrix==0)] = 0

        connections = node_connections.flatten().astype(int)
        obs = np.hstack([self.current_node, connections])
        if self.mask:
            cons_from_node = node_connections[self.current_node]
            mask = np.array([1 if c==1 and v==0 else 0 
                for c, v in zip(cons_from_node, self.visit_log.values())])
            state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.N),
                "state": obs,
            }
        else:
            state = obs.copy()

        return state
        
    def _generate_connections(self):
        node_dict = {}
        for n in range(self.N):
            connections = np.random.randint(2, self.N - 1)
            node_dict[n] = np.sort(
               np.random.choice(self.nodes[np.where(self.nodes!=n)],
                                 size=connections, replace=False))
            
        # TODO: Ensure at least two connections for each node to avoid
        # dead-ends.
        # Get unique, bi-directional connections
        for k, v in node_dict.items():
            for k1, v1 in node_dict.items():
                if k == k1:
                    continue
                if k in v1 and k1 not in v:
                    v = np.append(v, k1)

            node_dict[k] = np.sort(v.copy())
        self.node_dict = deepcopy(node_dict)
        self._generate_connection_matrix()
    
    def _generate_connection_matrix(self):
        self.connection_matrix = np.zeros((self.N, self.N))
        for k, v in self.node_dict.items():
            self.connection_matrix[k][v] += 1
        self.connection_matrix.astype(int)
            
    def _generate_coordinates(self):
        n = np.linspace(0, 2*np.pi, self.N+1)
        x = np.cos(n)
        y = np.sin(n)
        return np.vstack([x, y])

    def _get_node_distance(self, N0, N1):
        return np.sqrt(np.power(N0[0] - N1[0], 2) + np.power(N0[1] - N1[1], 2))
            
    def plot_network(self, offset=(0.02, 0.02)):
        coords = self._generate_coordinates()
        fig, ax = plt.subplots(figsize=(12,8))
        ax.scatter(coords[0], coords[1], s=40)
        for n, c in self.node_dict.items():
            for k in c:
                line = np.vstack([coords[:, n], coords[:, k]])
                dis = self._get_node_distance(line[0], line[1])
                # dis = np.sqrt(np.power(line[0, 0] - line[1, 0], 2) + 
                #               np.power(line[0, 1] - line[1, 1], 2))
                ax.plot(line[:,0], line[:,1], c='g', zorder=-1)
        #         ax.arrow(line[0, 0], line[0, 1], line[1, 0], line[1, 1])
            ax.annotate(r"$N_{:d}$".format(n), xy=(line[0]+offset), zorder=2)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.show()