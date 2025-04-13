# TXAI - Active Learning using Uncertainty Estimation: calculate mean and std of final model accuracies
# Group 20: Jiri Derks and Martijn van der Meer

import numpy as np

# Input accuracies of MC Dropout models w/ variation ratio and calculate mean and std
final_accuracies_MCD_varR = [84.26, 84.14, 83.65, 83.62, 84.26]

mean_accuracy = np.mean(final_accuracies_MCD_varR)
std_accuracy = np.std(final_accuracies_MCD_varR)
print("MCD_varR accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}\n")

# Input accuracies of MC Dropout models w/ variance and calculate mean and std
final_accuracies_MCD_variance= [59.30, 76.13, 75.06, 80.23, 79.44]

mean_accuracy = np.mean(final_accuracies_MCD_variance)
std_accuracy = np.std(final_accuracies_MCD_variance)
print("MCD_variance accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}\n")

# Input accuracies of Ensemble models w/ variation ratio and calculate mean and std
final_accuracies_ENS_varR = [87.94,88.31,87.78,87.94,88.23,87.35,88.08,87.78,87.46,87.98]
mean_accuracy = np.mean(final_accuracies_ENS_varR)
std_accuracy = np.std(final_accuracies_ENS_varR)
print("ENS_varR accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}\n")

# Input accuracies of Ensemble models w/ variance and calculate mean and std
final_accuracies_ENS_variation = [86.96,86.46,87.73,87.6,87.05,87.44,87.27,87.35,87.31,87.5]
mean_accuracy = np.mean(final_accuracies_ENS_variation)
std_accuracy = np.std(final_accuracies_ENS_variation)
print("ENS_variation accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}\n")

# Input accuracies of Dropconnect models w/ variance and calculate mean and std
final_accuracies_MCConnect_variance = [73.58, 80.73, 68.70, 69.40, 74.26]
mean_accuracy = np.mean(final_accuracies_MCConnect_variance)
std_accuracy = np.std(final_accuracies_MCConnect_variance)
print("MCConnect_variance accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}\n")

# Input accuracies of Dropconnect models w/ variation ratio and calculate mean and std
final_accuracies_MCConnect_varR = [85.29,85.85,76.09,85.14,86.04]
mean_accuracy = np.mean(final_accuracies_MCConnect_varR)
std_accuracy = np.std(final_accuracies_MCConnect_varR)
print("MCConnect_varR accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}\n")