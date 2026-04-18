import random
import torch.nn as nn

from metaqnn.config.rl_config import *
from metaqnn.config.train_config import *

from metaqnn.state_actions import get_action_values, to_string, parse_state
from metaqnn.state_actions import load_Q, save_Q, load_buffer, save_buffer

from metaqnn.train import train, initialize_datasets, create_model, get_optimizer


def q_learning(num_episodes, Q_file_path=None, buffer_file_path=None):
    # Initialize Q and replay buffer
    Q = load_Q(Q_file_path)
    replay_buffer = load_buffer(buffer_file_path)

    # Initialize datasets
    train_loader, val_loader, _ = initialize_datasets()

    for _ in range(num_episodes):
        S, U = sample_new_network(Q, epsilon=0.3)   # TODO: update epsilon
        model = create_model(S)
        optimizer = get_optimizer(model)

        accuracy = train(
            model, num_epochs=NUM_EPOCHS, train_loader=train_loader, val_loader=val_loader,
            loss_func=nn.CrossEntropyLoss(), optimizer=optimizer, scheduler=None
        )

        replay_buffer.append((S, U, accuracy))

        for _ in range(REPLAY_NUMBER):
            # Sample from replay buffer
            S_sample, U_sample, accuracy_sample = random.choice(replay_buffer)

            # Update Q values
            Q = update_Q_values(Q, S_sample, U_sample, accuracy_sample)


def sample_new_network(Q, epsilon):
    # Initialize state and action sequences
    state_sequence = [None]
    action_sequence = []

    while True:
        rand = random.random()

        if rand > epsilon:
            # Take the greedy action
            possible_actions, action_values = get_action_values(Q, state_sequence[-1])

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
            possible_actions, _ = get_action_values(Q, state_sequence[-1])
            rand_action = random.randint(0, len(possible_actions)-1)

            next_layer = possible_actions[rand_action]

        action_sequence.append(next_layer)

        if next_layer['layer_type'] == TERMINATION:
            break

        state_sequence.append(next_layer)

    return state_sequence, action_sequence


def update_Q_values(Q, S, U, accuracy):
    last_state = to_string(S[-1])
    last_action = to_string(U[-1])

    Q[last_state][last_action] = (1 - ALPHA) * Q.get(last_state, {}).get(last_action, INITIAL_Q_VALUE) + ALPHA * accuracy

    next_state = last_state
    for i in range(len(S) - 2, -1, -1):
        state = to_string(S[i])
        action = to_string(U[i])

        # Find best action for next state
        best_action_value = 0
        for next_action in Q[next_state]:   # TODO: not all actions will have been initialized ?
            best_action_value = max(best_action_value, Q[next_state][next_action])

        Q[state][action] = (1 - ALPHA) * Q.get(state, {}).get(action, INITIAL_Q_VALUE) + ALPHA * best_action_value

        next_state = state

    # Save Q

    return Q


if __name__ == "__main__":
