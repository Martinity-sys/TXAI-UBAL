import numpy as np

final_accuracies_MCD_varR = [84.26,84.14,83.65,83.62,84.26]

mean_accuracy = np.mean(final_accuracies_MCD_varR)
std_accuracy = np.std(final_accuracies_MCD_varR)
print("MCD_varR accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}")

final_accuracies_MCD_variation = [59.30,76.13,75.06,80.23,79.44]

mean_accuracy = np.mean(final_accuracies_MCD_variation)
std_accuracy = np.std(final_accuracies_MCD_variation)
print("MCD_variation accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}")