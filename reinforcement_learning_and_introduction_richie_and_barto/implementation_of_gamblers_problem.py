import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

# set the goal of the game
GOAL = 100

# create all states in the game
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.4


def figure_4_3():
    state_values = np.zeros(GOAL + 1)
    state_values[GOAL] = 1
    
    sweep_history = []
    
    # value iteration
    while True:
        old_state_values = state_values.copy()
        sweep_history.append(old_state_values)
        
        for state in STATES[1:GOAL]:
            # get possible actions for current state
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:
                action_returns.append(HEAD_PROB * state_values[state + action] + (1 - HEAD_PROB) * state_values[state - action])
                
            new_value = np.max(action_returns)
            state_values[state] = new_value
            
        delta = abs(old_state_values - state_values).max()
        if delta < 1e-9:
            sweep_history.append(state_values)
            break
        
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        
        for action in actions:
            action_returns.append(HEAD_PROB * state_values[state + action] + (1 - HEAD_PROB) * state_values[state - action])
        
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) +  1]
        
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweep_history):
        plt.plot(state_value, label=f"sweep {sweep}")
    plt.xlabel("Capital")
    plt.ylabel("Value Estimates")
    plt.legend(loc="best")
    
    
    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel("Capital")
    plt.ylabel("Final Policy (stake)")
    
    plt.savefig("figure_4_3.png")
    plt.close()



if __name__ == "__main__":
    figure_4_3()