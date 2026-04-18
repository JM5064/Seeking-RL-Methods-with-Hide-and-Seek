import json
from metaqnn.config.rl_config import *
from metaqnn.config.train_config import *


def get_action_values(Q, state):
    """Gets all actions for a given state and their associated values"""

    possible_actions = []

    # If start state, initialize state so the following lines don't crash lol
    if state is None:
        state = { 'layer_type' : None, 'layer_depth' : 0, 'representation_size': IMAGE_SIZE }

    layer_type = state['layer_type']
    layer_depth = state['layer_depth']
    representation_size = state['representation_size']

    if layer_depth >= MAX_DEPTH:
        # If at max depth, must go to terminal state
        possible_actions.append({ 'layer_type' : TERMINATION })

    elif layer_type is None:
        # If initial state, go to convolution, pooling, or FC
        possible_actions.extend(get_convolution_actions(layer_depth=layer_depth, representation_size=representation_size))
        possible_actions.extend(get_pooling_actions(layer_depth=layer_depth, representation_size=representation_size))
        possible_actions.extend(get_fully_connected_actions(layer_depth=layer_depth, num_consecutive=0))

    elif layer_type == CONVOLUTION:
        # Convolution layers can go to any layer
        possible_actions.extend(get_convolution_actions(layer_depth=layer_depth, representation_size=representation_size))
        possible_actions.extend(get_pooling_actions(layer_depth=layer_depth, representation_size=representation_size))
        possible_actions.extend(get_fully_connected_actions(layer_depth=layer_depth, num_consecutive=0))
        possible_actions.append({ 'layer_type' : TERMINATION })

    elif layer_type == POOLING:
        # Pooling layers can go to convolution, FC, or terminal
        possible_actions.extend(get_convolution_actions(layer_depth=layer_depth, representation_size=representation_size))
        possible_actions.extend(get_pooling_actions(layer_depth=layer_depth, representation_size=representation_size))
        possible_actions.append({ 'layer_type' : TERMINATION })

    elif layer_type == FULLY_CONNECTED:
        # FC layers can go to FC or terminal
        possible_actions.extend(get_fully_connected_actions(layer_depth=layer_depth, num_consecutive=state['num_consecutive'], curr_num_neurons=state['num_neurons']))
        possible_actions.append({ 'layer_type' : TERMINATION })

    else:
        print("should not happen prolly unless we decide to implement GAP")

    # Get action values for each state-action pair (or default if not yet explored)
    state_string = to_string(state)
    action_values = [
        # Get action value by converting state to string
        Q.get(state_string, {}).get(to_string(action), INITIAL_Q_VALUE)
        for action in possible_actions
    ]

    return possible_actions, action_values


def get_convolution_actions(layer_depth, representation_size):
    convolution_actions = []
    
    # Build all convolution types
    for num_channels in AVAIL_NUM_CHANNELS:
        for kernel_size in AVAIL_KERNEL_SIZES:
            # Only allow kernel_size < curr_representation_size
            if kernel_size >= representation_size:
                continue

            convolution_actions.append({ 
                'layer_type' : CONVOLUTION, 
                'out_channels' : num_channels, 
                'kernel_size' : kernel_size,
                'layer_depth' : layer_depth + 1,
                'representation_size': representation_size
            })

    return convolution_actions


def get_pooling_actions(layer_depth, representation_size):
    pooling_actions = []

    # Build all pooling types:
    for (kernel_size, stride) in AVAIL_KERNEL_SIZE_STRIDES:
        # Only allow kernel_size < curr_representation_size
        if kernel_size >= representation_size:
            continue

        pooling_actions.append({ 
            'layer_type' : POOLING, 
            'kernel_size' : kernel_size, 
            'stride' : stride,
            'layer_depth' : layer_depth + 1,
            'representation_size': (representation_size - kernel_size) // stride + 1
        })

    return pooling_actions


def get_fully_connected_actions(num_consecutive, layer_depth, curr_num_neurons=None):
    # If already at max consecutive FC, no fully connected actions available
    if num_consecutive >= MAX_CONSECUTIVE_FC:
        return []
    
    fully_connected_actions = []

    # Build all FC types:
    for num_neurons in AVAIL_NUM_NEURONS:
        # Only allow neurons <= number of current neurons
        if curr_num_neurons and num_neurons > curr_num_neurons:
            continue

        fully_connected_actions.append({ 
            'layer_type' : FULLY_CONNECTED, 
            'num_neurons' : num_neurons, 
            'num_consecutive' : num_consecutive + 1,
            'layer_depth' : layer_depth + 1,
            'representation_size': 1
        })

    return fully_connected_actions


def save_Q(Q, Q_file_path):
    """
    Q looks like:
    dict {
        "{ 'layer_type' : 0, ... }" : 
            {
                "{ 'layer_type' : 0, ... }" : 0.5,
                "{ 'layer_type' : 0, ... }" : 0.6,
                "{ 'layer_type' : 0, ... }" : 0.3
            }
        ,
        "{ 'layer_type' : 0, ... }" : 
            {
                "{ 'layer_type' : 0, ... }" : 0.5,
                "{ 'layer_type' : 0, ... }" : 0.6,
                "{ 'layer_type' : 0, ... }" : 0.3
            }
        ...
    }
    """
    with open(Q_file_path, 'w') as file:
        json.dump(Q, file, indent=4)


def load_Q(Q_file_path):
    with open(Q_file_path) as file:
        Q = json.load(file)

    return Q


def save_buffer(replay_buffer):
    pass


def load_buffer():
    pass


def to_string(state):
    return json.dumps(state)


def parse_state(state_string):
    return json.loads(state_string)


if __name__ == "__main__":
    Q = {
        "{'layer_type': 'convolution', 'out_channels' : 16, 'kernel_size' : 3}" : {
            "{'layer_type': 'convolution', 'out_channels' : 16, 'kernel_size' : 3}" : 0.4,
            "{'layer_type': 'convolution', 'out_channels' : 16, 'kernel_size' : 5}" : 0.2,
        },

        "{'layer_type': 'convolution', 'out_channels' : 16, 'kernel_size' : 5}" : {
            "{'layer_type': 'convolution', 'out_channels' : 16, 'kernel_size' : 3}" : 0.5,
            "{'layer_type': 'convolution', 'out_channels' : 16, 'kernel_size' : 5}" : 0.6
        }
    }

    save_Q(Q, 'metaqnn/saves/Q_values.json')