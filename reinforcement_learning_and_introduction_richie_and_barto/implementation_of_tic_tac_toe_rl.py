import numpy as np
import pickle

'''
This is the implementation of the tic tac toe  example
with a temporal difference learning algorithm which 
does ont have a model
'''

# specify number of rows and columns in the tic tac toe board
NUM_ROWS = 3
NUM_COLUMNS = 3
BOARD_SIZE = NUM_COLUMNS * NUM_ROWS


# A class used to represent one instance of state in the game, i.e and arrangement of x's and o's on the board
class State:
    def __init__(self):
        # 1 represents player 1
        # -1 represents player 2
        # 0 represents an empty solution
        self.data = np.zeros((NUM_COLUMNS, NUM_COLUMNS))
        self.winner = None # represents the winner in this state
        self.hash_val = None # represents the hash value of this state which is going to be important when we want to build the value table
        self.end = None # represents if the state is an end state
    
    # this is a function to generate a hash value for the state
    def hash(self):
        if not self.hash_val:
            self.hash_val = 0
            # iterate through the data currently in the matrix to generate state hash
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1
        
        # return the hash value     
        return self.hash_val
    
    # checks if a player has won the game or its a tie
    def is_end(self):
        
        # if we have already computed an end value then simply return it
        if self.end:
            return self.end
        
        results = []
        
        # check for results in rows
        for i in range(NUM_ROWS):
            # calculate the sum of values row wise i.e left to right
            results.append(np.sum(self.data[i, :])) # row vector
            
        for i in range(NUM_COLUMNS):
            # calculate the sum of values column wise i.e top to bottom
            results.append(np.sum(self.data[:,i])) # column vector
            
        # check diagonals
        traces = 0
        reverse_traces = 0
        
        for i in range(NUM_ROWS):
            # forward diagonal
            traces += self.data[i, i]
            # backward diagonal
            reverse_traces += self.data[i, (NUM_ROWS-1) -i]
            
        # add diagonals to result list
        results.append(traces)
        results.append(reverse_traces)
        
        # loop through all the results to see if there is any winning line
        for result in results:
            # if a result line has 3 then it means that it has all 1's which means 1 won
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            
            # if a result line has -3 then it means that it has all -1's which means -1 won
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end
            
        # check if its a draw
        sum_of_values = np.sum(np.abs(self.data))
        if sum_of_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end
        
        # game is still going
        self.end = False
        return self.end
    
    # generate the next state after putting symbol in a particular position
    # symbol can either be 1 for player 1 or -1 for player 2
    def next_state(self, i, j, symbol):
        # create a new state object
        next_state = State()
        # copy the content of the current state data into the new state object
        next_state.data = self.data.copy()
        # set the symbol on the location specified in the parameter
        next_state.data[i, j] = symbol
        # return the new state object
        return next_state
    
    # print the board of the current state
    def print_board(self):
        print(' ')
        for i in range(NUM_ROWS):
            print('-------------')
            out = "| "
            for j in range(NUM_COLUMNS):
                if self.data[i, j] == 1:
                    token = "X"
                elif self.data[i, j] == -1:
                    token = "O"
                else:
                    token = " "
                out += token + " | "
            print(out)
        print('-------------')
        print(' ')
        
        
# This generates board states for all possible states in the game
# The algorithm is a bit confusing, but it uses recursion to generate all possible states
def get_all_state_impl(current_state: State, current_symbol, all_states):
    for i in range(NUM_ROWS):
        for j in range(NUM_COLUMNS):
           if current_state.data[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
                    if not is_end:
                        get_all_state_impl(new_state, -current_symbol, all_states)
                    
             
# This will return hash map containing all the states in the game       
def get_all_states():
    current_symbol = 1
    current_state = State()
    # current_state.print_board()
    all_states = dict() # look up table we use to store all our states
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_state_impl(current_state, current_symbol, all_states)
    return all_states

all_states = get_all_states()

print("The number of possible states are: ", len(all_states))


class Player:
    # step_size: the amount of which to adjust estimations
    # epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        
        self.greedy = []
        self.states = []
        self.symbol = 0
    
    # reset the agents state
    def reset(self):
        self.greedy = []
        self.states = []
    
    # set the current state of the current state by adding it to the list of states
    # the agent acts greedily by default
    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)
        
    
    # sets the symbol of the player and and create the estimation hash table
    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_value in all_states:
            state, is_end = all_states[hash_value]
            if is_end and state.winner == symbol:
                self.estimations[hash_value] = 1.0
            elif is_end and state.winner != symbol:
                self.estimations[hash_value] = 0.0
            else:
                self.estimations[hash_value] = 0.5
    
    # update the estimations based on value of future states
    def backup(self):
        # create a list of all states the agent has been throgh
        states = [state.hash() for state in self.states]
        
        # move back through the states updating the values of the estimation based on the estimation of the next state by a factor of the step size
        # for values that do not follow the greedy algorithm the td error should be flipped to its negative value
        for i in reversed(range(len(states) - 1)):
            td_error = self.greedy[i] * (self.estimations[states[i+1]] - self.estimations[states[i]])
            self.estimations[states[i]] += td_error * self.step_size
    
    # chose action based on state, i.e policy implementation
    def act(self):
        state:State = self.states[-1] # the current state
        next_states = [] # the list of all possible next state positions
        next_actions = []  # the list of all possible next actions
        
        # get all possible next positions based on our current state
        for i in range(NUM_ROWS):
            for j in range(NUM_COLUMNS):
                # loop through all positions and see all possible single step acitons we can take to be in a new state
                if state.data[i, j] == 0:
                    next_actions.append([i, j]) # add action to our list of possible actions
                    next_states.append(state.next_state(i, j, self.symbol).hash()) # add state to our list of possible states

        # print("The number of next actions is: ",len(next_actions))
        # print("The number of next states is: ",len(next_states))
        # if(len(next_actions) == 0):
        #     state.print_board()
        
        # if the random value is less than the epsilon value then act randomly
        if np.random.rand() < self.epsilon:
            # randomly select action from list of possible actions, this is exploration
            action = next_actions[np.random.randint(len(next_actions))]
            action.append(self.symbol)
            # set this step as not greedy as the action was not greedily selected
            self.greedy[-1] = False
            return action
        
        else:
            values = []
            for (action, state) in zip(next_actions, next_states):
                values.append((self.estimations[state], action))
            
            # randomly shuffle values to select alternate action of equal value to avoid exploitation
            np.random.shuffle(values)
            
            # sort the values in reverse order to get the action with the highest value
            values.sort(key=lambda x: x[0], reverse=True)
            # print(values)
            action = values[0][1]
            # add the symbol to the aciton to be taken
            action.append(self.symbol)
            return action
    
    # save generated policy 
    def save_policy(self):
        if self.symbol == 1:
            name = "first"
        else:
            name = "second"
        
        with open(f"policy_{name}.bin", "wb") as f:
            pickle.dump(self.estimations, f)
    
    # load previously created policy
    def load_policy(self):
        if self.symbol == 1:
            name = "first"
        else:
            name = "second"
        
        with open(f"policy_{name}.bin", "rb") as f:
            self.estimations = pickle.load(f)
        print("Policy loaded successfully")
            

class Judger:
    # player1 is the player that moves first
    # player2 is the player that moves second
    
    def __init__(self, player1, player2):
        # initiate the player one and player 2
        self.pl1:Player = player1
        self.pl2:Player = player2
        
        # set the current player to None as the game is still initiating
        self.current_player = None
        
        # set the symbols for player 1 and player 2
        self.pl1_symbol = 1
        self.pl2_symbol = -1
        
        # set the symbols of player 1 and player 2
        self.pl1.set_symbol(self.pl1_symbol)
        self.pl2.set_symbol(self.pl2_symbol)
        self.current_state = State()
        
    
    # reset player 1 and 2 i.e removing all previously stored states and greedy actions
    def reset(self):
        self.pl1.reset()
        self.pl2.reset()
        
    # alternatively choose between player 1 and player 2 using yield
    def alternate(self):
        while True:
            yield self.pl1
            yield self.pl2
    
    # implementation of the game play loop
    def play(self, print_state=False, print_last=False):
        # initiat the alternator
        alternator  = self.alternate()
        
        # since a new game is beginning reset the game and the players
        self.reset()
        
        # the current state should be an empty board
        current_state:State = State()
        self.pl1.set_state(current_state)
        self.pl2.set_state(current_state)
        
        if print_state:
            current_state.print_board()
        
        while True:
            player = next(alternator)
            i, j, symbol = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            current_state, is_end = all_states[next_state_hash]
            
            self.pl1.set_state(current_state)
            self.pl2.set_state(current_state)
            if print_state:
                current_state.print_board()
            
            if is_end:
                print("Game ended...")
                if print_last:
                    current_state.print_board() # print board if game has ended irrespective of the value of print_state
                return current_state.winner
            

# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state:State = None
        
    def reset(self):
        pass
    
    def set_state(self, state):
        self.state = state
        
    def set_symbol(self, symbol):
        self.symbol = symbol
    
    def act(self):
        self.state.print_board()
        key = input()
        data = self.keys.index(key)
        i = data // NUM_ROWS
        j = data % NUM_COLUMNS
        
        return i, j, self.symbol
    

def train(epochs, print_every=500):
    # initialize player agents
    player1 = Player(0.01)
    player2 = Player(0.01)
    
    judger = Judger(player1, player2)
    player1_wins = 0
    player2_wins = 0
    
    for i in range(1, epochs+1):
        winner = judger.play()
        if winner == player1.symbol:
            player1_wins += 1
        if winner == player2.symbol:
            player2_wins += 1
        
        # show the stats every 500 games
        if i % print_every == 0:
            p1_win_rate = player1_wins/i
            p2_win_rate = player2_wins/i
            print(f"Player 1 has won {p1_win_rate} percent of the games and player 2 has won {p2_win_rate} percent of the games")
        # backup the agent to updates its estimations i.e the value function
        player1.backup()
        player2.backup()
        
        # reset game for next play
        judger.reset()
        
    # save the policy of both agents
    player1.save_policy()
    player2.save_policy()
    
    
def compute(turns):
    # create player 1 and two
    player1 = Player()
    player2 = Player()
    # set up a game judger
    judger =  Judger(player1, player2)
    # load previous policies from training
    player1.load_policy()
    player2.load_policy()
    player1_wins = 0
    player2_wins = 0
    
    for i in range(turns):
        winner = judger.play()
        if winner == player1.symbol:
            player1_wins += 1
        if winner == player2.symbol:
            player2_wins += 1
        judger.reset()
    
    p1_win_rate = player1_wins/turns
    p2_win_rate = player2_wins/turns
    
    print(f"Player 1 win rate is {p1_win_rate} and Player 2 win rate is {p2_win_rate}")
    

# AI against human play
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player()
        
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play(print_last=True)
        
        if winner == player1.symbol:
            print("You win!")
            
        if winner == player2.symbol:
            print("You lose!")
            
            
if __name__ == "__main__":
    # train(int(1e5))
    # compute(int(1e3))
    play()
            
            
            
            
        
            