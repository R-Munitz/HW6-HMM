import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        """
        pseudocode - based on textbook
        create probability matrix forward, dimensions [N, T]
        N = len of state graph, T = len of observations
        for each state s in range (1, N):  #initialization
            forward[s, 1] = prior[s] (pi s) * emission[s, observations[1]] (bs(o1))
        for each time step t in range (2, T):  #recursion
            for each state s in range (1, N):
                forward[s, t] = sum(forward[s', t-1] * transition[s', s] * emission[s, observations[t]]) for each state s'
        forward_probability = sum(forward[s, T]) for each state s
        return forward_probability
        """
    
        # Step 1. Initialize variables
        N = len(self.hidden_states)
        T = len(input_observation_states)

        # step 1.1 - edge cases
        #empy input sequence
        if T == 0:
            #raise error
            raise ValueError("Input sequence is empty!")
        
        #single observation input
        if T == 1:
            forward_probability = sum([
            self.prior_p[state] * self.emission_p[state, self.observation_states_dict[input_observation_states[0]]]
            for state in range(N)
            ])
            return forward_probability
    
        forward_probability_mat = np.zeros((N, T))
        
        # Step 2. Calculate probabilities
        for state in range(N):
            forward_probability_mat[state, 0] = self.prior_p[state] * self.emission_p[state, self.observation_states_dict[input_observation_states[0]]]
        for time_step in range(1, T): #1 = 2, ix starts at 0
            for state in range(N):
                forward_probability_mat[state, time_step] = sum([forward_probability_mat[state_prime, time_step-1] * self.transition_p[state_prime, state] * self.emission_p[state, self.observation_states_dict[input_observation_states[time_step]]] 
                                                                 for state_prime in range(N)]) 
            forward_probability = sum([forward_probability_mat[state, T-1] for state in range(N)])
           
        # Step 3. Return final probability 
        return forward_probability        

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """       

        """
        #pseudocode - from textbook
        N = len of state graph, T = len of observations
        create a path probability aka viterbi table, dimensions [N, T]
        for each state s in range (1, N):  #initialization
            viterbi[s,1] = prior_p [s] (pi[s] ) * emission[s, observations[1]] (bs(o1))
            backpointer[s, 1] = 0
        for each time step t in range (2, T):  #recursion
            for each state s in range (1, N):
                viterbi_table[s,t] = max(viterbi_table[s', t-1] * transition_p[s', s] * emission_p[s, observations[t]]) for each state s'
                backpointer[s,t] = argmax(viterbi_table[s', t-1] * transition_p[s', s]) * emission_p[s, observations[t]]) for each state s'
        best_path_prob = max(viterbi_table[s, T]) for each state s
        best_path_pointer = argmax(viterbi_table[s, T]) for each state s
        best_path = [best_path_pointer]
        trace back to find best path
        return best_path, best_path_prob
        """ 
        
        
        # Step 1. Initialize variables
        

        N = len(self.hidden_states)
        T = len(decode_observation_states)

        #Step 1.1 - edge cases

        #empy input sequence
        if T == 0:
            #raise error
            raise ValueError("Input sequence is empty!")
        
        #single observation input
        if T == 1:
            best_path_pointer = np.argmax([
                self.prior_p[state] * self.emission_p[state, self.observation_states_dict[decode_observation_states[0]]] 
                for state in range(N)
                ])
            return self.hidden_states_dict[best_path_pointer]
            
        viterbi_table = np.zeros((N, T))

        #store best path pointers for traceback
        best_path = np.zeros((N, T), dtype=int)

        for state in range(N):
            viterbi_table[state][0] = self.prior_p[state] * self.emission_p[state, self.observation_states_dict[decode_observation_states[0]]]
            best_path[state][0] = 0
        
        for time_step in range(1, T):
            for state in range(N):
                viterbi_table[state][time_step] = max([viterbi_table[state_prime][time_step -1]* self.transition_p[state_prime][state] * self.emission_p[state, self.observation_states_dict[decode_observation_states[time_step]]] 
                                                       for state_prime in range(N)])
                best_path[state][time_step] = max(
                    range(N), 
                    key=lambda prev_state: viterbi_table[prev_state][time_step-1] * self.transition_p[prev_state][state])

        best_path_pointer = max(range(N),
                key=lambda state: viterbi_table[state][T-1])

        best_path_list = [best_path_pointer]

        #traceback to find best path
        for time_step in range(T-1, 0, -1):  # start from the last time step going backward
            best_path_list.insert(0, best_path[best_path_list[0], time_step])

        #decode best path using dict 
        best_hidden_state_sequence = [self.hidden_states_dict[state] for state in best_path_list]
 
        
        return best_hidden_state_sequence
                