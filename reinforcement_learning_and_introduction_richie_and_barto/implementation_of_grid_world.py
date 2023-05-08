# This is the implementation of the grid world example in the chapter 3 of the reinforcement learning book
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.table import Table


matplotlib.use("Agg")


WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]

DISCOUNT = 0.9

ACTIONS = [
    np.array([0, -1]),
    np.array([-1, 0]),
    np.array([0, 1]),
    np.array([1, 0])]

ACTIONS_FIGS=[ '←', '↑', '→', '↓']

ACTION_PROB = 0.25


def step(state, action):
    # This is the function to take a single step
    # in the grid world, it returs the new position and the reward
    
    # if the makes a move in A_POS they are teleported to A_PRIME and given a reward of 10
    if state == A_POS:
        return A_PRIME_POS, 10
    
    # if the user steps into B_POS they are teleported to B_PRIEM and given a reward of 5
    if state == B_POS:
        return B_PRIME_POS, 5
    
    # create the new state as 0, 0 position and then construct 
    # the new state by adding the action to the current state
    new_state = [0, 0]
    new_state[0] = state[0] + action[0]
    new_state[1] = state[1] + action[1]
    
    # if the state is out of bound  return the old state with a reward of -1
    if new_state[0] >= WORLD_SIZE or new_state[1] >= WORLD_SIZE or new_state[0] < 0 or new_state[1] < 0:
        return state, -1
    
    # if the new position is valid move to the new position with a reward of 0
    else:
        return new_state, 0


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    
    nrows, ncols = image.shape
    width, height = 1.0/ncols, 1.0/nrows
    
    # add cells
    for (i, j), val in np.ndenumerate(image):
        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A*)"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B*)"

        
        tb.add_cell(i, j, width, height, text=val, loc="center", facecolor="white")
    
    
    for i in range(len(image)):
    # row and column labels
        if i in range(len(image)):
            tb.add_cell(i, -1, width, height,  text=str(i+1), loc="right", edgecolor="none", facecolor="none")
            tb.add_cell(-1, i, width, height/2, text=str(i+1), loc="center", edgecolor="none", facecolor="none")
            
    ax.add_table(tb)


def draw_policy(optimal_values):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    
    
    nrows, ncols = optimal_values.shape
    width, height = 1.0/ncols, 1.0/nrows
    
    # add cells
    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals = []
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0], next_state[1]])
            
        best_actions = np.where(next_vals == np.max(next_vals))[0]
        val = ''
        
        for ba in best_actions:
            val += ACTIONS_FIGS[ba]
        
        # add state label
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A*)"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B*)"
            
        tb.add_cell(i, j, width, height, text=val, loc="center", facecolor="white")
        
        # Row and Column labels
        for i in range(len(optimal_values)):
            tb.add_cell(i, -1, width, height, text=i+1, loc="right", edgecolor="none", facecolor="none")
            tb.add_cell(-1, i, width, height/2, text=i+1, loc="center", edgecolor="none", facecolor="none")

        ax.add_table(tb)
        

def figure_3_2():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration till convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
                    
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig("figure_3_2.png")
            plt.close()
            break
        value = new_value
                     
        
def figure_3_2_linear_system():
    """
    We solve the linear system of equation and fill each cell with its appropriate value, using an equiprobabilistic policy 
    for all the actions
    """
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j] # current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in ACTIONS:
                s_, r = step(s, a)
                index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))
                
                
                A[index_s, index_s_] += ACTION_PROB * DISCOUNT
                b[index_s] -= ACTION_PROB * r
                
    x = np.linalg.solve(A, b)
    draw_image(np.round(x.reshape(WORLD_SIZE, WORLD_SIZE), decimals=2))
    plt.savefig("figure_3_2_lienar_system.png")
    plt.close()
    
    

def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration till convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + DISCOUNT* (value[next_i, next_j]))

                new_value[i, j] = np.max(values)
        

        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig("figure_3_5.png")
            plt.close()
            draw_policy(new_value)
            plt.savefig("figure_3_5_policy.png")
            plt.close()
            break
        
        value = new_value
        
        
if __name__ == "__main__":
    # figure_3_2()
    # figure_3_2_linear_system()
    figure_3_5()