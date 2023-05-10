# import required packages
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import tqdm
import itertools
from functools import partial
import multiprocessing as mp
import time


# specify problem specific contstants
MAX_CARS = 20
MAX_MOVE = 5
MOVE_COST = -2
ADDITIONAL_PARK_COST = -4

RENT_REWARD = 10

# set up requests for the two locations
REQUEST_LOC1 = 3
REQUEST_LOC2 = 4

# set up returns for the two locations
RETURNS_LOC1 = 3
RETURNS_LOC2 = 2

poisson_cache = dict()

def poisson(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = math.exp(-lam) * math.pow(lam, n) / math.factorial(n)
    return poisson_cache[key]


class PolicyIteration:
    def __init__(self, truncate, parallel_processes, delta=1e-2, gamma=0.9, solve_4_5=False):
        # initialize policy iteration class
        self.TRUNCATE = truncate
        self.NP_PARALLEL_PROCESSES = parallel_processes
        self.actions = np.arange(-MAX_CARS, MAX_CARS+1)
        self.inverse_actions = {el: ind[0] for ind, el in np.ndenumerate(self.actions)}
        self.values =  np.zeros((MAX_CARS+1, MAX_CARS+1))
        self.policy = np.zeros(self.values.shape, dtype=np.int64)
        self.delta = delta
        self.gamma = gamma
        self.solve_extension = solve_4_5
        
    # solve the policy iteration
    def solve(self):
        iterations = 0
        total_start_time = time.time()
        
        while True:
            start_time = time.time()
            self.values = self.policy_evaluation(self.values, self.policy)
            elapsed_time = time.time() - start_time
            print(f"ITERATION {iterations+1} Time elapsed for policy evalution is : {elapsed_time} seconds")
            
            start_time = time.time()
            policy_change, self.policy = self.policy_improvement(self.actions, self.values, self.policy)
            elapsed_time = time.time() - start_time
            print(f"ITERATION {iterations+1} Time elapsed for policy impromvement is : {elapsed_time} seconds")
            
            if policy_change == 0:
                break
            
            iterations += 1
        
        elapsed_time = time.time() - total_start_time
        print(f"Total time for policy iteration and improvement is : {elapsed_time}, for total number of iterations {iterations}")
        
    
    def policy_evaluation(self, values, policy):
        
        global MAX_CARS
        while True:
            new_values = np.copy(values)
            k = np.arange(MAX_CARS + 1)
            # cartesian product
            all_states = ((i, j) for i, j in itertools.product(k, k))
            
            results = []
            with mp.Pool(processes=self.NP_PARALLEL_PROCESSES) as p:
                cook = partial(self.expected_return_pe, policy, values)
                results = p.map(cook, all_states)
                
            for v, i, j in results:
                new_values[i, j] = v
            
            difference = np.abs(new_values - values).sum()
            print("Difference: ", difference)
            values = new_values
            
            # if the change in values is less than delta, then the values have sufficiently converged
            if difference < self.delta:
                print("Values have converged")
                return values
            
            
    def policy_improvement(self, actions, values, policy):
        new_policy = policy.copy()
        expected_action_returns = np.zeros((MAX_CARS+1, MAX_CARS+1, np.size(actions)))
        cooks = dict()
        
        with mp.Pool(processes=8) as p:
            for action in actions:
                k = np.arange(MAX_CARS + 1)
                all_states = ((i, j) for i, j in itertools.product(k, k))
                cooks[action] = partial(self.expected_return_pi, values, action)
                results = p.map(cooks[action], all_states)
                for v, i, j, a in results:
                    expected_action_returns[i, j, self.inverse_actions[a]] = v
                    
        for i in range(expected_action_returns.shape[0]):
            for j in range(expected_action_returns.shape[1]):
                new_policy[i, j] = actions[np.argmax(expected_action_returns[i, j])]
        
        policy_change = (new_policy != policy).sum()
        print("Policy changed in {policy_change} states")
        return policy_change, new_policy
    
    
    
    def bellman(self, values, action, state):
        expected_return = 0
        if self.solve_extension:
            if action > 0:
                # free shuttle to the second location
                expected_return += MOVE_COST * (action - 1)
            else:
                expected_return += MOVE_COST * abs(action)
                
        else:
            expected_return += MOVE_COST * abs(action)
            
        for req1 in range(0, self.TRUNCATE):
            for req2 in range(0, self.TRUNCATE):
                # moving cars
                num_cars_loc1 = int(min(state[0] - action, MAX_CARS))
                num_cars_loc2 = int(min(state[1] + action, MAX_CARS))
                
                # valid rental requests should be less than the nunmber of cars available at the booth
                real_rental_loc1 = min(num_cars_loc1, req1)
                real_rental_loc2 =  min(num_cars_loc2, req2)
                
                # total credits gained for successfully renting
                reward = (real_rental_loc1 + real_rental_loc2) * RENT_REWARD
                
                if self.solve_extension:
                    if num_cars_loc1 >= 10:
                        reward += ADDITIONAL_PARK_COST
                    if num_cars_loc2 >= 10:
                        reward += ADDITIONAL_PARK_COST

                num_cars_loc1 -= real_rental_loc1
                num_cars_loc2 -= real_rental_loc2
                
                prob = poisson(req1, REQUEST_LOC1) * poisson(req2, REQUEST_LOC2)
                
                for ret1 in range(0, self.TRUNCATE):
                    for ret2 in range(0, self.TRUNCATE):
                        num_cars_loc1_ = min(num_cars_loc1 + ret1, MAX_CARS)
                        num_cars_loc2_ = min(num_cars_loc2 + ret2, MAX_CARS)
                        
                        prob_ = poisson(ret1, RETURNS_LOC1) * poisson(ret2, RETURNS_LOC2) * prob
                        expected_return += prob_ * (reward + self.gamma * values[num_cars_loc1_,  num_cars_loc2_])
                        
        return expected_return
    
    # parallelization requires different helper mehtods
    # expected return calculator for policy evaluation
    def expected_return_pe(self, policy, values, state):
        action = policy[state[0], state[1]]
        expected_return = self.bellman(values, action, state)
        return expected_return, state[0], state[1]
    
    # greedy action improvement based on values
    def expected_return_pi(self, values, action, state):
        if ((action >= 0 and state[0] >= action) or (action < 0 and state[1] >= abs(action))) == False:
            return -float('inf'), state[0], state[1], action
        else:
            return self.bellman(values, action, state), state[0], state[1], action
        
        
    def plot(self):
        print(self.policy)
        plt.figure()
        plt.xlim(0, MAX_CARS + 1)
        plt.ylim(0, MAX_CARS + 1)
        plt.table(cellText=np.flipud(self.policy), loc=(0, 0), cellLoc="center")
        plt.show()
        

if __name__ == "__main__":
    TRUNCATE = 9
    solver = PolicyIteration(TRUNCATE, parallel_processes=4, delta=1e-1, gamma=0.9, solve_4_5=True)
    solver.solve()
    solver.plot()
        
            
                    
                
            
            
            
        
        
        


