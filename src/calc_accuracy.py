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

final_accuracies_ENS_varR = [88.04,
    87.5,
    87.88,
    87.72,
    87.94,
    88.31,
    87.78,
    87.94,
    88.23,
    87.35,
    88.08,
    87.78,
    87.46,
    87.98]
mean_accuracy = np.mean(final_accuracies_ENS_varR)
std_accuracy = np.std(final_accuracies_ENS_varR)
print("ENS_varR accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}")

final_accuracies_ENS_variation = [87.49,
86.96,
86.46,
87.73,
87.6,
87.05,
87.44,
87.27,
87.35,
87.31,
87.5]
mean_accuracy = np.mean(final_accuracies_ENS_variation)
std_accuracy = np.std(final_accuracies_ENS_variation)
print("ENS_variation accuracies:")
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}")