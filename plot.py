import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file without headers
data = np.genfromtxt('accuracy_batch.csv', delimiter=',')
data2 = np.genfromtxt('accuracy.csv', delimiter=',')

# Generate x-axis values (each datapoint corresponds to 40 samples)
x_values = np.arange(1,len(data)+1) * 40  # Scaling indices by 40

# Plot the values
plt.plot(x_values, data, label='Accuracy_batch')
plt.plot(x_values, data2, label='Accuracy')

# Add labels and title
plt.xlabel('Samples Processed')
plt.ylabel('Accuracy')
plt.title('Accuracy over Samples')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
