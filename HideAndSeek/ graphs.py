import matplotlib.pyplot as plt
import numpy as np


def get_episode_metrics(monitor_csv):
    rewards = []
    losses = []
    times = []

    prev_time = 0

    # Open file
    with open(monitor_csv) as file:
        for i, item in enumerate(file):
            if i <= 1:
                continue

            split = item.split(",")

            reward = float(split[0])
            loss = float(split[1])
            time = float(split[2])

            rewards.append(reward)
            losses.append(loss)
            if i == 2:
                times.append(time)
            else:
                times.append(time-prev_time)

            prev_time = time

            # print(times[-1])

    return rewards, losses, times


def plot_rewards(base_monitor_csv, metaqnn_monitor_csv):
    base_rewards, base_losses, base_times = get_episode_metrics(base_monitor_csv)
    metaqnn_rewards, metaqnn_losses, metaqnn_times = get_episode_metrics(metaqnn_monitor_csv)

    plt.plot(base_rewards)
    plt.plot(metaqnn_rewards)
    plt.show()


def plot_barcharts():
    models = ['Base Model', 'MetaQNN']
    avg_rewards = [-0.6816999725298956, -0.5172399583365768]
    avg_timesteps = [54.05, 81.46]

    x = np.arange(len(models))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, avg_rewards, width, label='Avg Reward per Episode')
    plt.bar(x + width/2, avg_timesteps, width, label='Avg Time Steps per Episode')

    plt.xticks(x, models)
    plt.ylabel('Values')
    plt.title('Model Comparison')
    plt.legend()

    plt.show()

    # Base model:
    # Avg reward per episode: -0.6816999725298956
    # Avg reward per timestep: -0.012612395421459678
    # Avg time steps per episode: 54.05

    # MetaQNN:
    # Avg reward per episode: -0.5172399583365768
    # Avg reward per timestep: -0.006349618933667774
    # Avg time steps per episode: 81.46 





if __name__ == "__main__":
    base_monitor_csv = 'HideAndSeek/logs_base_model/monitor.csv'
    metaqnn_monitor_csv = 'HideAndSeek/logs_Meta_QNN_model/monitor.csv'

    # plot_rewards(base_monitor_csv, metaqnn_monitor_csv)
    plot_barcharts()
