import json
import numpy as np
import matplotlib.pyplot as plt


def find_top_k(log_file_path, k=5):
    # open file
    with open(log_file_path) as file:
        log = json.load(file)

    log.sort(key=lambda model : -model[2])

    return log[:k]


def get_rolling_means(log_file_path, average_over=4):
    # open file
    with open(log_file_path) as file:
        log = json.load(file)

    accuracies = [accuracy for _, _, accuracy in log]

    rolling_means = np.convolve(accuracies, np.ones(average_over)/average_over, mode='valid')

    return rolling_means


def get_average_accuracy_per_epsilon(log_file_path, epsilon):
    # open file
    with open(log_file_path) as file:
        log = json.load(file)
    
    sum_accuracy = 0
    total_epsilon = 0

    for _, e, accuracy in log:
        if e != epsilon:
            continue

        sum_accuracy += accuracy
        total_epsilon += 1

    return sum_accuracy / total_epsilon


def plot_q_learning_performance(log_file_path):
    # open file
    with open(log_file_path) as file:
        log = json.load(file)

    n = len(log)
    epsilons = [epsilon for _, epsilon, _ in log]

    # Find epsilon change points
    change_points = [0]
    for i in range(1, n):
        if epsilons[i] != epsilons[i-1]:
            change_points.append(i)
    change_points.append(n)

    plt.figure(figsize=(8, 6))

    # Plot epsilon averages
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i+1]
        epsilon = epsilons[start]

        average_accuracy = get_average_accuracy_per_epsilon(log_file_path, epsilon)

        plt.fill_between(
            range(start, end),
            average_accuracy,
            alpha=0.25,
            color='tab:blue',
            label="Average Accuracy Per Epsilon" if i == 0 else None
        )

        # Label epsilon
        epsilon_text = str(epsilon)
        if start == 0:
            epsilon_text = "Epsilon = " + epsilon_text

        plt.text(x=(start + end) // 2, y=0.02, s=epsilon_text, ha='center', fontsize=10)

    # Plot rolling mean line
    rolling_means = get_rolling_means(log_file_path)
    plt.plot(rolling_means, label="Rolling Mean Model Accuracy")
    plt.ylim(0, 1)

    # Labels / title
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Q-Learning Performance")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    log_file_path = 'metaqnn/logs/logs.json'

    top_k = find_top_k(log_file_path, k=6)

    for model in top_k:
        architecture, epsilon, accuracy = model
        for layer in architecture:
            print(layer)
        print(epsilon)
        print(accuracy)
        print()

    plot_q_learning_performance(log_file_path)