import numpy as np
from hmm import HiddenMarkovModel 

def load_data():
    """Load HMM parameters and observation sequences from .npz files."""
    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')

    print("Available keys in mini_weather_hmm.npz:", mini_hmm.files)  # Debugging line


    print("Available keys in mini_weather_sequences.npz:", mini_input.files)  # Debugging line


    # Extract parameters
    hidden_states = mini_hmm["hidden_states"]
    observation_states = mini_hmm["observation_states"]
    prior_p = mini_hmm["prior_p"]
    transition_p = mini_hmm["transition_p"]
    emission_p = mini_hmm["emission_p"]
    
    observation_state_sequence = mini_input["observation_state_sequence"]
    print("Observation State Sequence:", observation_state_sequence)  # Debugging line
    best_hidden_state_sequence = mini_input["best_hidden_state_sequence"]
    print("Best Hidden State Sequence:", best_hidden_state_sequence)  # Debugging line

    return hidden_states, observation_states, transition_p, emission_p, prior_p, observation_state_sequence, best_hidden_state_sequence


def main():
    """Runs the HMM on test sequences and prints output for debugging."""
    
    # Load HMM data
    hidden_states, observation_states, transition_p, emission_p, prior_p, observation_state_sequence, best_hidden_state_sequence = load_data()
    
    # Instantiate the HMM
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)
    
    # Test sequences
    test_seqs = observation_state_sequence

    print("\n=== Running Forward Algorithm ===")
    forward_probs = hmm.forward(test_seqs)
    print("Forward Probabilities Table:\n", forward_probs)

    print("\n=== Running Viterbi Algorithm ===")
    viterbi_path = hmm.viterbi(test_seqs)
    print("Viterbi Best Path:", viterbi_path)

    print("expected:", best_hidden_state_sequence)

    # Validate outputs
    print("\n=== Validating Results ===")
    if np.array_equal(viterbi_path, best_hidden_state_sequence):
        print(" Viterbi output matches expected!")
    else:
        print(" Viterbi output does NOT match expected!")

    # Edge Cases
    print("\n=== Running Edge Cases ===")

    # Edge Case 1: Empty input sequence
    empty_result = hmm.viterbi([])
    print("Viterbi Output for Empty Sequence:", empty_result)

    # Edge Case 2: Single observation input
    single_obs_result = hmm.viterbi([test_seqs[0]])
    single_forward_probs = hmm.forward([test_seqs[0]])
    print("Viterbi Output for Single Observation:", single_obs_result)
    print("Viterbi output:", single_forward_probs)

if __name__ == "__main__":

    main()
