import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_lambda_curves(csv_file):
    """
    Reads lambda values from a CSV file and plots them.
    - First epoch (bold blue), last epoch (bold red).
    - Intermediate epochs use a high-contrast colormap and different line styles.
    """
    # Load CSV file
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)

    # Convert data to float while handling NaN and -inf
    data = []
    for line in lines[1:]:  # Skip header
        cleaned_values = [
            float(x.replace("[", "").replace("]", "")) if "nan" not in x and "-inf" not in x else 0.0 
            for x in line
        ]
        data.append(cleaned_values)

    data = np.array(data)  
    num_epochs, num_lambdas = data.shape  # Get dimensions

    # Use a strong contrast colormap (viridis)
    colors = plt.cm.viridis(np.linspace(0, 1, num_epochs))  # Distinct colors for each curve
    linestyles = ['-', '--', '-.', ':']  # Different line styles

    plt.figure(figsize=(10, 8))

    # Plot all lambda curves
    for i in range(num_epochs):
        linestyle = linestyles[i % len(linestyles)]  # Cycle through line styles
        
        if i == 0:
            plt.plot(data[i], color="blue", linewidth=2.5, linestyle='-', label="Initial sigmoid")  
        elif i == num_epochs - 1:
            plt.plot(data[i], color="red", linewidth=2.5, linestyle='-', label="Last Epoch")  
        else:
            plt.plot(data[i], color=colors[i], linestyle=linestyle, alpha=0.8, label = f"Epoch {i}")

    plt.xlabel("Lambda Index")
    plt.ylabel("Lambda Value")
    plt.title("Lambda Values Over Training Epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

def plot_losses_from_csv(csv_file):
    """
    Reads train and test losses from a CSV file and plots them.
    The CSV file is expected to have a header: "Train Loss,Test Loss"
    with two columns of loss values.
    """
    train_losses = []
    test_losses = []

    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:  # Ensure there are at least two columns
                try:
                    train_losses.append(float(row[0]))
                    test_losses.append(float(row[1]))
                except ValueError:
                    continue  # Skip invalid rows

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o', color='b')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", marker='s', color='r')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Testing Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    lambda_log_file = "GNNS_project/code/GNNS_final_project/logs/lambda_log_small.csv"
    training_log_file = "GNNS_project/code/GNNS_final_project/logs/training_log_small.csv"

    plot_lambda_curves(lambda_log_file)
    plot_losses_from_csv(training_log_file)
