import numpy as np

def read_viterbi_input(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    sections = [section.strip() for section in ''.join(data).split('%%') if section.strip()]

    if len(sections) != 4:
        raise ValueError(f"File format error: expected 4 sections, but found {len(sections)}.")

    try:
        obs_data = [float(num.strip()) for num in sections[0].split('\n') if num]
        init_data = [float(num.strip()) for num in sections[1].split('\n') if num]
        transition_data = [float(num.strip()) for num in sections[2].split('\n') if num]
        emission_data = [float(num.strip()) for num in sections[3].split('\n') if num]
    except Exception as e:
        raise ValueError(f"Error parsing data: {e}")

    # Compute N_STATES by taking the square root of the length of the transition data
    N_STATES = len(init_data)
    
    # Check if N_STATES is valid
    if N_STATES * N_STATES != len(transition_data):
        raise ValueError(f"Invalid transition matrix size: expected {N_STATES * N_STATES} elements, got {len(transition_data)}.")

    N_TOKENS = len(emission_data) // N_STATES
    N_OBS = len(obs_data)

    # Reshape data into matrices and arrays
    transition = np.array(transition_data).reshape((N_STATES, N_STATES))
    emission = np.array(emission_data).reshape((N_STATES, N_TOKENS))
    init = np.array(init_data)
    obs = np.array(obs_data, dtype=int)

    return init, transition, emission, obs

# file_path = 'input.data'
# try:
#     init, transition, emission, obs = read_viterbi_input(file_path)

#     # Print the parsed sections for verification
#     print("Initial probabilities (log-space):\n", init)
#     print("Transition matrix (log-space):\n", transition)
#     print("Emission matrix (log-space):\n", emission)
#     print("Observations:\n", obs)
# except ValueError as e:
#     print(f"An error occurred: {e}")
