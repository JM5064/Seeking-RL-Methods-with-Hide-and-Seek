import random
import torch.nn as nn
import json

from metaqnn.config.rl_config import *
from metaqnn.config.train_config import *

from metaqnn.state_actions import get_possible_actions, get_action_values, to_string, parse_state
from metaqnn.state_actions import load_Q, save_Q, load_buffer, save_buffer

from metaqnn.qlearning import update_Q_values
from metaqnn.train import train, initialize_datasets, create_model, get_optimizer, get_scheduler
from metaqnn.log import save_model_metrics


Q_file_path = 'metaqnn/logs/Q_values_restarted.json'
buffer_file_path = 'metaqnn/logs/replay_buffer_restarted.pkl'
log_json_path = 'metaqnn/logs/logs.json'


def q_learning(num_episodes, start_episode=0):
    # Initialize Q and replay buffer
    Q = load_Q(Q_file_path)
    replay_buffer = load_buffer(buffer_file_path)

    for episode in range(start_episode, num_episodes):
        # Fast forward
        S, U, accuracy = parse_SU_accuracy(log_json_path, episode)

        replay_buffer.append((S, U, accuracy))

        # for _ in range(min(len(replay_buffer) * 10, REPLAY_NUMBER)):
        #     # Sample from replay buffer
        #     S_sample, U_sample, accuracy_sample = random.choice(replay_buffer)

        #     # Update Q values
        #     Q = update_Q_values(Q, S_sample, U_sample, accuracy_sample)

        weights = [m[2]**2 for m in replay_buffer] 

        # 2. Sample K models based on those weights
        samples = random.choices(replay_buffer, weights=weights, k=min(len(replay_buffer) * 10, REPLAY_NUMBER))

        for S_sample, U_sample, accuracy_sample in samples:
            Q = update_Q_values(Q, S_sample, U_sample, accuracy_sample)

        # Save new Q values and buffer
        save_Q(Q, Q_file_path)
        save_buffer(replay_buffer, buffer_file_path)


def parse_SU_accuracy(log_path, model_num):
    # open file
    with open(log_path) as file:
        log = json.load(file)

    S = [None]
    U = []

    architecture, epsilon, accuracy = log[model_num]

    for i in range(len(architecture)):
        U.append(architecture[i])

        if i != len(architecture) - 1:
            S.append(architecture[i])

    return S, U, accuracy

if __name__ == "__main__":
    q_learning(num_episodes=190, start_episode=0)
    # buffer = load_buffer(buffer_file_path)
    # q_learning(num_episodes=300, start_episode=len(buffer))
