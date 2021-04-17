import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils import assign_env_config
from copy import copy

class PortfolioOptEnv(gym.Env): 
    '''
    Portfolio Optimization Problem 

    Instance: Multi-Period Asset Allocation Problem, Dantzing & Infager, 1993 

    The Portfolio Optimization (PO) Problem is a problem that seeks to optimize 
    the distribution of assets in a financial portfolio to with respect to a desired 
    financial metric (e.g. maximal return, minimum risk, etc.). 

    In this particular instance by Dantzing & Infager, the optimizer begins with a 
    quantity of cash and has the opportunity to purchase or sell 3 other assets in each  
    of 10 different investement periods. Each transaction incurs a cost and prices of 
    the 3 assets are subject to change over time. Cash value is consant (price = 1). 
    The objective is to maximize the amount of wealth (i.e. the sum total of asset values)
    at the end of the total investment horizon.

    The episodes proceed by the optimizer deciding whether to buy or sell each asset
    in each time period. The episode ends when either all 10 periods have passed or 
    if the amount of any given asset held becomes negative.  

    Observation:
        Type: Box(9)
        "asset prices" (idx 0, 1, 2, 3): array of asset prices [cash, asset1, asset2, asset3]
        "asset quantities" (idx 4, 5, 6, 7): array of asset quantities [cash, asset1, asset2, asset3]
        "total wealth" (idx 8): current total wealth (sum of price*quantity for each asset)
    

    Actions:
        Type: Box (3)
        "asset 1 transaction amount" (idx 0): x in [-2000, 2000]: Buy (positive) or sell (negative) x shares of asset 1; 
        "asset 2 transaction amount" (idx 1): x in [-2000, 2000]: Buy (positive) or sell (negative) x shares of asset 2; 
        "asset 3 transaction amount" (idx 2): x in [-2000, 2000]: Buy (positive) or sell (negative) x shares of asset 3; 

    Reward:
        Change in total wealth from previous period or [-max(asset price of all assets) *  maximum transaction size]
        if an asset quantity becomes negative, at which 
        point the episode ends.

    Starting State:
        Starting amount of cash and wealth and prices. 

    Episode Termination:
        Negative asset quantity or traversal of investment horizon. 
    '''    
    def __init__(self, *args, **kwargs):
        self.num_assets = 3 # Number of assets 
        self.initial_cash = 100 # Starting amount of capital 
        self.step_limit = 10 # Investment horizon 

        self.cash = copy(self.initial_cash)

        #Transaction costs proportional to amount bought 
        self.buy_cost = np.array([0.045, 0.025, 0.035])
        self.sell_cost = np.array([0.04, 0.02, 0.03])
        # self.step_limit = 10
        # assign_env_config(self, kwargs)
        # self.asset_price_means = asset_price_means.T
        # # self.asset_price_means = (np.random.randint(10, 50, self.num_assets) \
        # #     * np.ones((self.step_limit, self.num_assets))).T
        # self.asset_price_var = np.ones(self.asset_price_means.shape)

        # Prices of assets have a mean value in every period and vary according to a Gaussian distribution 
        asset1mean = np.array([1.25, 2, 4, 5, 3, 2, 3, 6, 9, 7]).reshape(1, -1)  # Up and down all the way 
        asset2mean = np.array([5, 3, 2, 2, 1.25, 4, 5, 6, 7, 8]).reshape(1, -1)  # Down intially then up 
        asset3mean = np.array([3, 5, 6, 9, 10, 8, 4, 2, 1.25, 4]).reshape(1, -1) # Up initially then down 
        self.asset_price_means = np.vstack([asset1mean, asset2mean, asset3mean])
        self.asset_price_var = np.ones((self.asset_price_means.shape)) * 0.45
        
        # Cash on hand, asset prices, num of shares, portfolio value
        self.obs_length = 1 + 2 * self.num_assets

        self.observation_space = spaces.Box(-20000, 20000, shape=(self.obs_length,))
        self.action_space = spaces.Box(-2000, 2000, shape=(self.num_assets,))
        
        self.seed()
        self.reset()
        
        
    def _RESET(self):
        self.step_count = 0
        self.asset_prices = self._generate_asset_prices()
        self.holdings = np.zeros(self.num_assets)
        self.cash = copy(self.initial_cash)
        self.state = np.hstack([
            self.initial_cash,
            self.asset_prices[:, self.step_count],
            self.holdings])
        return self.state
    
    def _generate_asset_prices(self):
        asset_prices = np.array([self.np_random.normal(mu, sig) for mu, sig in 
            zip(self.asset_price_means.flatten(), self.asset_price_var.flatten())]
            ).reshape(self.asset_price_means.shape)
        # Zero out negative asset prices and all following prices - implies
        # equity is bankrupt and worthless.
        zero_vals = np.vstack(np.where(asset_prices<0))
        cols = np.unique(zero_vals[0])
        for c in cols:
            first_zero = zero_vals[1][np.where(zero_vals[0]==c)[0].min()]
            asset_prices[c,first_zero:] = 0
        return asset_prices
    
    def _STEP(self, action):
        
        assert self.action_space.contains(action)
    
        asset_prices = self.asset_prices[:, self.step_count].copy()
        
        for idx, a in enumerate(action):
            if a == 0:
                continue
            # Sell a shares of asset
            elif a < 0:
                a = np.abs(a)
                if a > self.holdings[idx]:
                    a = self.holdings[idx]
                self.holdings[idx] -= a
                self.cash += asset_prices[idx] * a * (1 - self.sell_cost[idx])
            # Buy a shares of asset
            elif a > 0:
                purchase_cost = asset_prices[idx] * a * (1 + self.buy_cost[idx])
                if self.cash < purchase_cost:
                    a = np.floor(self.cash / (
                        asset_prices[idx] * (1 + self.buy_cost[idx])))
                    purchase_cost = asset_prices[idx] * a * (1 + self.buy_cost[idx])
                self.holdings[idx] += a
                self.cash -= purchase_cost
                
        # Return total portfolio value at the end of the horizon as reward
        if self.step_count + 1 == self.step_limit: 
            reward = np.dot(asset_prices, self.holdings) + self.cash
        else: 
            reward = 0 
        self.step_count += 1

        # Finish if 10 periods have passed - end of investment horizon 
        if self.step_count >= self.step_limit:
            done = True
        else:
            self._update_state()
            done = False
            
        return self.state, reward, done, {}
    
    def _update_state(self):
        self.state = np.hstack([
            self.cash,
            self.asset_prices[:, self.step_count],
            self.holdings
        ])

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
