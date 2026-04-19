import os


def save_accuracy(epsilon, accuracy, file_path):
    "Saves epsilon, accuracy into csv file"
    # Open file
    file = open(file_path, "a")

    # Create header if file is blank
    if os.path.getsize(file_path) == 0:
        file.write(f'epsilon,accuracy')
        
        file.write('\n')

    # Log metrics
    file.write(f'{epsilon},{accuracy},\n')

    # Makes file update immediately
    file.flush()
    os.fsync(file.fileno())
