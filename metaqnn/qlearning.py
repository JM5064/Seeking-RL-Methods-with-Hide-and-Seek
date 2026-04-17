import random
from metaqnn.rl_config import *
from metaqnn.train_config import *
from metaqnn.state_actions import *

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
    action_values = [
        Q.get(state, {}).get(action, INITIAL_Q_VALUE)
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
                'representation_size': (representation_size - kernel_size) + 1
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


def sample_new_network(epsilon, Q):
    # Initialize state and action sequences
    state_sequence = [None]
    layers = [None]

    while True:
        rand = random.random()

        if rand > epsilon:
            # Take the greedy action
            possible_actions, action_values = get_action_values(Q, layers[-1])

            # TODO: make this faster
            # TODO: make it pick a random one if tie
            best_action = None
            best_action_value = 0
            for i in range(len(possible_actions)):
                if action_values[i] > best_action_value:
                    best_action = possible_actions[i]
                    best_action_value = action_values[i]

            next_layer = best_action

        else:
            # Take a random action
            possible_actions, _ = get_action_values(Q, layers[-1])
            rand_action = random.randint(0, len(possible_actions)-1)

            next_layer = possible_actions[rand_action]

        state_sequence.append(next_layer)
        layers.append(next_layer)

        if next_layer['layer_type'] == TERMINATION:
            break



def init_Q():
    # Load Q if stored ?

    Q = {}

    return Q