# implemenation of the 10 armed bandit test bed
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib
from typing import List


matplotlib.use("Agg")

class Bandit:
    # A single bandit problem with k arms
    # @k_arm : number of arms for the agent to select from
    # @epsilon : exploration rate (probability to select a non greedy solution at every given time step)
    # @initial : the initial value for hte bandit problem, this is what the values for each arm will be initialized to
    # @sample_averages: If true, update using sample averages i.e 1/N(a) else use step size value
    # @UBC_param : this is the ubc update parameter 'c' in the ubc algorithm which is A = argmax [Q(a) + c * sqrt(log(t)/N(a)))]
    # @gradient: if true use gradient bandit based algorithm
    # @gradient_baseline: this is the initial baseline used in gradient bandit based algorithm
    # @true_reward: The value fo the reward
    def __init__(self, k_arm=10, epsilon=0, initial=0, step_size=0.1, sample_averages=False, 
                 UCB_param=None, gradient=False, gradient_baseline=False, true_reward=0):
        self.k = k_arm
        self.epsilon = epsilon
        self.initial = initial
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.true_reward = true_reward
        # set the initial time step value
        self.time = 0
        # set the initial average rewarc
        self.average_reward = 0
        # set the indices and their indexes for all the arms
        self.indices = np.arange(self.k)
        
    # reset the bandit to the beginning
    def reset(self):
        # set the real reward for each action
        self.q_true_karms = np.random.randn(self.k) + self.true_reward
        # set up inital estimates for all the actions
        self.q_estimation_karms = np.zeros(self.k) + self.initial
        # action count to keep track of each time an action has been taken
        self.action_count = np.zeros(self.k)
        # set the inital best action
        self.best_action = np.argmax(self.q_true_karms)
        # reset time back to zero
        self.time = 0
    
    # set the q_true value
    def set_q_true(self, q_true):
        self.q_true_karms = q_true
        self.best_action = np.argmax(self.q_true_karms)

    
    # get an action from this bandit
    # @context is used to know what contex we are currently on using contextual bandits
    def act(self):
        # If the UCB param value is not none then we would be using the UCB technique
        if self.UCB_param is not None:
            # we would be calculating the UCB values for all our actions
            UCB_values = self.q_estimation_karms + (self.UCB_param * np.sqrt(np.log(self.time + 1)/self.action_count + 1e-5))
            # find the maximum ucb value
            UCB_max = np.max(UCB_values)
            # make a selection by randomly selecting from the list any value that has the highest UCB value. Note: np.where returns indexes
            selected_action = np.random.choice(np.where(UCB_values == UCB_max)[0])
            # return the selected action 
            return selected_action
        
        # if we are using the gradient method,then we select with a probability based on the softmax
        if self.gradient:
            # to scale down exponents as not to get too large we substract it by the max
            max_value = np.max(self.q_estimation_karms)
            # calculate the softmax probability with the scaled down values
            self.action_prob = np.exp(max_value - self.q_estimation_karms) / np.sum(np.exp(max_value - self.q_estimation_karms))
            # select an action from the indices baseed on the softmax probability distribution
            selected_action = np.random.choice(self.indices, p=self.action_prob)
            # return the selected action
            return selected_action
        
        # using the epsilone greedy strategy, if the epsilon value is 0 then the random generated action 
        # is never selected, because it can never generate less than zero
        if np.random.rand() < self.epsilon:
            # select a random choice from the indices
            selected_action = np.random.choice(self.indices)
            # return the selected action
            return selected_action
        
        # using the greedy strategy first find the value of the q_best
        q_best = np.max(self.q_estimation_karms)
        # select from the options of values that have the q_best
        selected_action = np.random.choice(np.where(self.q_estimation_karms == q_best)[0])
        # returntthe selected action
        return selected_action
    
    # take a step in the bandit problem
    def step(self, action):
        reward = np.random.randn() + self.q_true_karms[action]
        # increase the time step
        self.time+=1
        # increase the action count for the selected action
        self.action_count[action] += 1
        # update the averag reward based on the reward just received and using the total time as the step size
        self.average_reward += ((reward - self.average_reward)/self.time)
        
        # if using gradient technique
        if self.gradient:
            # create one hot encoding vector for the actions
            one_hot = np.zeros(self.k)
            # set the value of the selected action to 1 in the one hot encoding
            one_hot[action] = 1
            # set the baseline variable to the average reward if we are using gradient baseline methods
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                # set baseline to zero if we are not using gradient baseline methods
                baseline = 0
            # caclulate updates for all action value estimations using gradient ascent
            self.q_estimation_karms += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        
        # if using sample averages
        if self.sample_averages:
            self.q_estimation_karms[action] += ((reward - self.q_estimation_karms[action])/ self.action_count[action])
            
        # if using fixed step size
        self.q_estimation_karms[action] += self.step_size * (reward - self.q_estimation_karms[action])
        
        # return the reward
        return reward
    
    
    
class ContextualBandit:
    # a contextual bandit is composed of many individual bandits that are 
    # selected based on context
    # @number of contexts available
    # @k_arm : number of arms for the agent to select from
    # @epsilon : exploration rate (probability to select a non greedy solution at every given time step)
    # @initial : the initial value for hte bandit problem, this is what the values for each arm will be initialized to
    # @sample_averages: If true, update using sample averages i.e 1/N(a) else use step size value
    # @UBC_param : this is the ubc update parameter 'c' in the ubc algorithm which is A = argmax [Q(a) + c * sqrt(log(t)/N(a)))]
    # @gradient: if true use gradient bandit based algorithm
    # @gradient_baseline: this is the initial baseline used in gradient bandit based algorithm
    # @true_reward: The value fo the reward
    def __init__(self, num_contexts, k_arm=10, epsilon=0, initial=0, step_size=0.1, sample_averages=False, 
                 UCB_param=None, gradient=False, gradient_baseline=False, true_reward=0):
        
        # we are assuming that each bandit handles a different context so we will create 'num_contexts' bandits
        self.bandits:List[Bandit] = [Bandit(k_arm, epsilon, initial, step_size, sample_averages, UCB_param, gradient, gradient_baseline, true_reward) for context in range(num_contexts)]
        self.contexts = np.arange(num_contexts)
        self.time = 0
        self.average_reward = 0
        
    def reset(self):
        for bandit in self.bandits:
            bandit.reset()
    
    def act(self, context):
        return self.bandits[context].act()
    
    
    def best_action(self, context):
        return self.bandits[context].best_action
    
    def step(self, action, context):
        reward = self.bandits[context].step(action)
        self.time += 1
        self.average_reward += (reward - self.average_reward)/self.time
        return reward
        
    
    

def simulate(runs, time, bandits: List[Bandit]):
    # set all inital rewards to zero
    rewards = np.zeros((len(bandits), runs, time))
    # set all initial best action counts to zero
    best_action_count = np.zeros(rewards.shape)
    
    # loop through each bandit
    for b, bandit in enumerate(bandits):
        # loop through each run
        for r in trange(runs):
            # reset the bandit to a be a new distribution of values
            bandit.reset()
            # loop through each time step
            for t in range(time):
                # select action
                action = bandit.act()
                # move to the next step with action and set the reward for that particular time steppp
                rewards[b,r,t] = bandit.step(action)
                # the action is the best action then set the best action count of 1
                if action == bandit.best_action:
                    best_action_count[b, r, t] = 1
    
    # find the mean of selecting the best action for each timestep accross all the runs for each bandit, i.e all runs are compressed to one
    mean_best_action_count = best_action_count.mean(axis=1)
    # find the mean of the reward for each timestep accross all the runs for each bandit, i.e all runs are compressed to one
    mean_rewards = rewards.mean(axis=1)
    
    return mean_best_action_count, mean_rewards


def simulate_contextual_bandits(runs, time, bandits: List[ContextualBandit]):
    # set all inital rewards to zero
    rewards = np.zeros((len(bandits), runs, time))
    # set all initial best action counts to zero
    best_action_count = np.zeros(rewards.shape)
    
    # loop through each bandit
    for b, bandit in enumerate(bandits):
        # loop through each run
        for r in trange(runs):
            # reset the bandit to a be a new distribution of values
            bandit.reset()
            # loop through each time step
            for t in range(time):
                # select action
                context = np.random.choice(bandit.contexts)
                action = bandit.act(context)
                # move to the next step with action and set the reward for that particular time steppp
                rewards[b,r,t] = bandit.step(action, context)
                # the action is the best action then set the best action count of 1
                if action == bandit.best_action(context):
                    best_action_count[b, r, t] = 1
    
    # find the mean of selecting the best action for each timestep accross all the runs for each bandit, i.e all runs are compressed to one
    mean_best_action_count = best_action_count.mean(axis=1)
    # find the mean of the reward for each timestep accross all the runs for each bandit, i.e all runs are compressed to one
    mean_rewards = rewards.mean(axis=1)
    
    return mean_best_action_count, mean_rewards
    



def simulate_with_non_stationary_bandits(runs, time, bandits: List[Bandit]):
    # set the inital rewards to all zeros
    rewards = np.zeros((len(bandits), runs, time))
    # set the inital best action to all zeros
    best_action_count = np.zeros(rewards.shape)
    
    # loop through bandits
    for b, bandit in enumerate(bandits):
        # loop through runs
        for r in trange(runs):
            # create a shift point from 300 to the time specified
            shift_point = np.random.choice(np.arange(300, time-200))
            # create variable to check if the q true has been shifted
            shifted = False
            # create the shifted values
            shift_values = np.random.randn(bandit.k)
            # reset the bandit
            bandit.reset()
            # loop through the time steps
            for t in range(time):
                # if it is above the shift time and the value is not yet shifted, then we shift values
                if t > shift_point and not shifted:
                    # we set the q_value of the bandit to the shifted values
                    bandit.set_q_true(shift_values)
                    # we set the shift variable to true
                    shifted = True
                
                # the bandit picks an action
                action = bandit.act()
                # we set the reward based the action that was selected
                rewards[b, r, t] = bandit.step(action)
                # if the action selected was the best action, then we set the action selected count to 1
                if action == bandit.best_action:
                    best_action_count[b, r, t] = 1
                    
    # find the mean of selecting the best action for each timestep accross all the runs for each bandit, i.e all runs are compressed to one
    mean_best_action_count = best_action_count.mean(axis=1)
    # find the mean of the reward for each timestep accross all the runs for each bandit, i.e all runs are compressed to one
    mean_rewards = rewards.mean(axis=1)
    
    return mean_best_action_count, mean_rewards
            
                

def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.array([0.1, -0.9, 1.5, 0.4, 1.3, -1.5, -0.1, -1, 0.8, -0.5]))
    plt.xlabel("Action")
    plt.xticks(np.arange(1,11))
    plt.ylabel("Reward Distribution")
    plt.savefig('figure_2_1.png')
    plt.close()
    

def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.01, 0.1]
    bandits = [Bandit(epsilon=epsilon, sample_averages=True) for epsilon in epsilons]
    best_actions, rewards = simulate(runs, time, bandits)
    
    plt.figure(figsize=(10, 20))
    
    plt.subplot(2, 1, 1)
    for eps, reward in zip(bandits, rewards):
        plt.plot(reward, label='$\epsilon = %.02f$' % (eps.epsilon))
    
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    
    
    plt.subplot(2, 1, 2)
    for eps, best_action in zip(bandits, best_actions):
        plt.plot(best_action, label='$\epsilon = %.02f$' % (eps.epsilon))
    
    plt.xlabel("Steps")
    plt.ylabel("% optimal action")
    plt.legend()
    
    plt.savefig("figure_2_2.png")
    plt.close()
    
    

def exercise_2_5(runs=2000, time=10000):
    # create fixed step size 0.1 bandit with epsilon greedy of 0.1
    fixed_step_size_bandit = Bandit(epsilon=0.1, step_size=0.1)
    # create bandit with epsilon greedy of 0.1 and uses sample averages
    sample_avarage_bandit = Bandit(epsilon=0.1, sample_averages=True)
    bandits = [fixed_step_size_bandit, sample_avarage_bandit]
    
    # simulate in a regular situation
    best_actions, rewards = simulate(runs, time, bandits)
    plt.figure(figsize=(10, 20))
    
    plt.subplot(2, 1, 1)
    plt.plot(rewards[0], label="epsilon=0.1 fixed step size 0.1")
    plt.plot(rewards[1], label="epsilon=0.1 sample averages")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    
    # plt.subplot(2, 1, 2)
    # plt.plot(best_actions[0], label="epsilon=0.1 fixed step size 0.1")
    # plt.plot(best_actions[1], label="epsilon=0.1 sample averages")
    # plt.xlabel = "Steps"
    # plt.ylabel = "% optimal action"
    # plt.legend()
    
    
    
    best_actions, rewards = simulate_with_non_stationary_bandits(runs, time, bandits)
    plt.subplot(2, 1, 2)
    plt.plot(rewards[0], label="epsilon=0.1 fixed step size 0.1")
    plt.plot(rewards[1], label="epsilon=0.1 sample averages")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    
    # plt.subplot(2, 1, 4)
    # plt.plot(best_actions[0], label="epsilon=0.1 fixed step size 0.1")
    # plt.plot(best_actions[1], label="epsilon=0.1 sample averages")
    # plt.xlabel = "Steps"
    # plt.ylabel = "% optimal action"
    # plt.legend()
    
    plt.savefig("exercise_2_5.png")
    plt.close()
    
    
def exercise_2_10(runs=2000, time=1000):
    episolons = [0, 0.01, 0.1]
    bandits = [ContextualBandit(num_contexts=4, k_arm=4, epsilon=episolon, sample_averages=True) for episolon in episolons]
    
    best_actions, rewards = simulate_contextual_bandits(runs, time, bandits)
    
    plt.figure(figsize=(10, 20))
    
    
    plt.subplot(2, 1, 1)
    for eps, reward in zip(episolons, rewards):
        plt.plot(reward, label = f"epsilon={eps}")
    plt.xlabel("Average reward")
    plt.ylabel("Step")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for eps, best_action in zip(episolons, best_actions):
        plt.plot(best_action, label = f"epsilon={eps}")
    plt.xlabel("% best action")
    plt.ylabel("Step")
    plt.legend()
    
    
    plt.savefig("exercise_2_10.png")
    plt.close()
        
def figure_2_3(runs=2000, time=1000):
    # create list of bandits
    bandits = []
    # add bandit with no step size but optimistic start
    bandits.append(Bandit(epsilon=0, initial=5, step_size=0.1))
    bandits.append(Bandit(epsilon=0.1, initial=0, step_size=0.1))
    
    
    best_action, _ = simulate(runs, time, bandits)
    plt.plot(best_action[0], label="optimistic start with step size of 0.1")
    plt.plot(best_action[1], label="epsilon 0.1 with no optimistic start")
    
    plt.xlabel("steps")
    plt.ylabel("% best action")
    
    plt.legend()
    plt.title("Optimistic step size with epsilon 0 vs epsilon 0.1")
    
    plt.savefig("figure_2_3.png")
    plt.close()
    
    

if __name__ == '__main__':
    figure_2_3()
    
    




