import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_weather_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    
    mini_weather_hmm = HiddenMarkovModel(mini_weather_hmm['observation_states'], mini_weather_hmm['hidden_states'],
                                          mini_weather_hmm['prior_p'], mini_weather_hmm['transition_p'],
                                            mini_weather_hmm['emission_p'])
    
    #get forward probability for input sequence
    forward_prob = mini_weather_hmm.forward(mini_input['observation_state_sequence'])

    #assert forward probability matches expected value
    expected_forward_prob = 0.0351
    assert round(forward_prob, 4) == expected_forward_prob

    #get viterbi path for input sequence
    viterbi_path = mini_weather_hmm.viterbi(mini_input['observation_state_sequence'])

    #assert viterbi path matches expected path
    expected_viterbi_path = mini_input['best_hidden_state_sequence']
    assert np.array_equal(viterbi_path, expected_viterbi_path[0])

    #edge case 1: empty observation sequence
    empty_observation_sequence = []

    #assert value error raised
    with pytest.raises(ValueError):
        empty_forward_prob = mini_weather_hmm.forward(empty_observation_sequence)

    with pytest.raises(ValueError):
        viterbi_path = mini_weather_hmm.viterbi(empty_observation_sequence)


    #edge case 2: single observation sequence
    single_observation_sequence = ['sunny']
    single_observation_forward_prob = mini_weather_hmm.forward(single_observation_sequence)
    assert single_observation_forward_prob == 0.45
    single_viterbi_path = mini_weather_hmm.viterbi(single_observation_sequence)
    assert single_viterbi_path == "hot"

    pass



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_weather_hmm=np.load('./data/full_weather_hmm.npz')
    full_weather_input=np.load('./data/full_weather_sequences.npz')

    #forward probability
    forward_prob = full_weather_hmm.forward(full_weather_input['observation_state_sequence'])

    #viterbi path
    viterbi_path = full_weather_hmm.viterbi(full_weather_input['observation_state_sequence'])

    #assert viterbi path matches expected path
    expected_viterbi_path = full_weather_input['best_hidden_state_sequence']
    assert np.array_equal(viterbi_path, expected_viterbi_path[0])


    pass













