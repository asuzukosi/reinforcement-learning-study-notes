# This is the implementation of the car rental problem
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from scipy.stats import poisson

matplotlib.use("Agg")

# maximum number of cars in each location
MAX_CARS = 20
# maximum number of car moves
MAX_CAR_MOVES = 5

# number of requests in location 1
REQUEST_L1 = 3
# number of requests in location 2
REQUEST_L2 = 4

# number of returns in location 1
RETURN_L1 = 3
# number of returns in location 2
RETURN_L2 = 2

# discount rate
DISCOUNT = 0.9
# credits earned by moving a car
RENTAL_CREDITS = 10
# cost of moving a car
MOVE_CAR_COST = 2

# all possible actions
actions = np.arange(-MAX_CAR_MOVES, + MAX_CAR_MOVES+1)

# an upper bound for all distributions
POISSON_UPPER_BOUND = 11
# probability of poisson distribution
poisson_cache = dict()


def poisson_distribution(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key]  = poisson.pmf(n, lam)
    return poisson_cache[key]


def expected_return(state, action, state_value, constant_returned_cars):
    # The state is the number of cars in the first location and the second location
    # The action is positive for moving a car from location 1 to 2 and negative for moving a car from location 2 to 1
    # The state value is the state value matrix
    # The constant returned cars determines if we are using a constant amount of car stransfers or we are using 
    # a randomized poisson transfer
    
    # initialize total return
    returns = 0.0
    
    # the cost of moving a car
    returns -= MOVE_CAR_COST * np.abs(action)
    # moving cars
    NUM_CARS_IN_LOC1 = min(state[0] - action, MAX_CARS)
    NUM_CARS_IN_LOC2 = min(state[1] - action, MAX_CARS)
    
    # go through all the possible rental requests
    for rental_request_first_loc in range(POISSON_UPPER_BOUND):
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            # probabilty of current combination of rental requests
            prob = poisson_distribution(rental_request_first_loc, REQUEST_L1) * poisson_distribution(rental_request_second_loc, REQUEST_L2)
            
            num_cars_first_loc = NUM_CARS_IN_LOC1
            num_cars_second_loc = NUM_CARS_IN_LOC2
            
            # valid rental requests should be less the actuall number of cars in a specific location
            valid_rental_request_loc_1 = min(num_cars_first_loc, rental_request_first_loc)
            valid_rental_request_loc_2 = min(num_cars_second_loc, rental_request_second_loc)
            
            
            # get credits for successfully renting
            rewards = (valid_rental_request_loc_1 + valid_rental_request_loc_2) * RENTAL_CREDITS
            num_cars_first_loc -= valid_rental_request_loc_1
            num_cars_second_loc -= valid_rental_request_loc_2
            
            if constant_returned_cars:
                # get returned cars as tehy can be used for renting in the next time step
                returned_cars_loc1 = RETURN_L1
                returned_cars_loc2 = RETURN_L2
                
                num_cars_first_loc = min(returned_cars_loc1+num_cars_first_loc, MAX_CARS)
                num_cars_second_loc = min(returned_cars_loc2+num_cars_second_loc, MAX_CARS)
                returns += prob * (rewards + DISCOUNT * state_value[num_cars_first_loc, num_cars_second_loc])
            
            # in this case we would be using the poisson distribution 
            else:
                for returned_cars_l1 in range(POISSON_UPPER_BOUND):
                    for returned_cars_l2 in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_distribution(returned_cars_l1, RETURN_L1) * poisson_distribution(returned_cars_l2, RETURN_L2)
                        num_cars_first_loc_ = min(num_cars_first_loc+returned_cars_l1, MAX_CARS)
                        num_cars_second_loc_ = min(num_cars_second_loc+returned_cars_l2, MAX_CARS)
                        prob_ = prob_return * prob
                        returns += prob_ * (rewards + DISCOUNT * state_value[num_cars_first_loc_, num_cars_second_loc_])
                        
    return returns

                
                
                
def figure_4_2(constant_returned_cars=True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype=np.int64)

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)
        
        # policy evaluation (in-place)
        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max()
            print('max value change {}'.format(max_value_change))
            if max_value_change < 1e-4:
                break

        # policy improvement
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                action_returns = []
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-np.inf)
                new_action = actions[np.argmax(action_returns)]
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:
                    policy_stable = False
        print('policy stable {}'.format(policy_stable))
        
        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break
        
        iterations += 1
        
        
    plt.savefig("figure_4_2c.png")
    plt.close()

        
        
if __name__ == "__main__":
    figure_4_2(False)
        
    
            

