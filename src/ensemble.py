# TXAI - Active Learning using Uncertainty Estimation: Ensembles w/ Variation Ratio
# Simple CNN Ensemble on FMNIST
# Group 20: Jiri Derks and Martijn van der Meer

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchensemble import VotingClassifier


import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import numpy as np
import csv
import copy
import pickle
import argparse

########## Load Data

# Set Hyperparameters
argparser = argparse.ArgumentParser(description='Active Learning with Ensembles')
argparser.add_argument('--runs', type=int, default=10, help='number of runs')
argparser.add_argument('--save', type=bool, default=False, help='save model')
argparser.add_argument('--init', type=int, default=40, help='initial labeled set size')
argparser.add_argument('--acq', type=int, default=40, help='acquisition size')
argparser.add_argument('--max', type=int, default=2000, help='maximum labeled set size')
argparser.add_argument('--t', type=int, default=5, help='number of models in the ensemble')
argparser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
args = argparser.parse_args()

N_RUNS = args.runs
SAVE_MODEL = args.save
INIT_SIZE = args.init
ACQ_SIZE = args.acq
ACQ_MAX = args.max
NUM_EPOCHS = args.epochs
T = args.t



# Normalize images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create datasets for training & validation
full_training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Define loader for validation data, loader for training data is defined inside AL loop
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)

# Class labels
classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


########## Define model and uncertainty function


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Initialise some layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Define how the layers connect in a forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)

        return output

# Variation Ratio as measure of uncertainty    
def varR(predictions, T):

    res = []
    for sample in predictions:
        f_m = len(sample[sample == torch.mode(sample).values])
        res.append(1 - f_m/T)
    
    return torch.tensor(res)

########## Experiment Loop

# Initialise CSV writer to save metrics
f = open("data/model_data/dataENS_varR.csv", 'w', newline='')
writer = csv.writer(f)
writer.writerow(['run', 'train_size', 'Loss', 'Accuracy'])

for run in range(N_RUNS):

    # Define ensemble (using GPU if available)
    model = VotingClassifier(
        estimator = Net,
        n_estimators=T,
        cuda = torch.cuda.is_available(), 
    )

    # Set criterion
    criterion = nn.CrossEntropyLoss()

    # Select initial AL labeled indices list
    # Use list of indices of full train set to track current train set
    al_indices = torch.randint(high=len(full_training_set), size=(INIT_SIZE,))
    rem_indices = torch.tensor(range(0, len(full_training_set)))
    mask = np.ones(rem_indices.shape[0], dtype=bool)
    mask[al_indices] = False
    rem_indices = rem_indices[mask]


    # Create data loaders for our datasets; shuffle for training, not for validation
    # Improves data retrieval
    curr_train = torch.utils.data.Subset(full_training_set, al_indices)
    training_loader = torch.utils.data.DataLoader(curr_train, batch_size=64, shuffle=True)

    curr_rem = torch.utils.data.Subset(full_training_set, rem_indices)
    rem_loader = torch.utils.data.DataLoader(curr_rem, batch_size=512, shuffle=False)


    ############# AL loop

    # Train the model
    train_size = INIT_SIZE

    while(train_size <= ACQ_MAX):

        # Make copy of the model for current AL step
        curr_model = copy.deepcopy(model)
        curr_model.set_criterion(criterion)

        curr_model.set_optimizer(
            'Adam',
            lr = 0.001,
        )

        curr_model.fit(train_loader=training_loader, epochs = NUM_EPOCHS,log_interval=1000, save_model=False)

        # Calculate intermediate metrics and store in csv
        accuracy, loss = curr_model.evaluate(training_loader, return_loss=True)
        writer.writerow([run, train_size, loss, accuracy])

        # Calculate Uncertainty
        all_preds = torch.empty((0,T), dtype=torch.long, device="cuda")

        for i, (images, labels) in enumerate(rem_loader):

            batch_size = images.shape[0]
            curr_preds = torch.empty((batch_size, T), dtype=torch.float, device="cuda")

            for idx, model_ind in enumerate(curr_model.estimators_):

                images, labels = images.to("cuda"), labels.to("cuda")
                outputs = curr_model.estimators_[idx].forward(images)
                _, predicted = torch.max(outputs.data, 1)

                curr_preds[:, idx] = predicted
            all_preds = torch.cat((all_preds, curr_preds), dim=0)
                
        uncertainty = varR(all_preds, T)

        # Select n most uncertain samples and move samples to training set
        new_batch = torch.topk(uncertainty, k = ACQ_SIZE).indices
        al_indices = torch.cat((al_indices, rem_indices[new_batch]))
        mask = np.ones(rem_indices.shape[0], dtype=bool)
        mask[new_batch] = False
        rem_indices = rem_indices[mask]

        # Update data loaders
        curr_train = torch.utils.data.Subset(full_training_set, al_indices)
        training_loader = torch.utils.data.DataLoader(curr_train, batch_size=64, shuffle=True)

        curr_rem = torch.utils.data.Subset(full_training_set, rem_indices)
        rem_loader = torch.utils.data.DataLoader(curr_rem, batch_size=512, shuffle=False)

        # Update curr training size for while loop
        train_size += ACQ_SIZE

        if train_size > ACQ_MAX:
            model = curr_model

    print(f'Training run {run} complete!')

    if SAVE_MODEL:
        with open('./models/VarR/ensemble' + str(run) + '.nc', "wb") as f_final:
            pickle.dump(model, f_final)
        f_final.close()


    # Final metrics
    accuracy, loss = model.evaluate(validation_loader, return_loss=True)

    print(f'RUN {run}: Loss of the trained model on the test images: {loss:.4f}')
    print(f'RUN {run}: Accuracy of the trained model on the test images: {100 * accuracy:.2f}%')


    #save average accuracy and loss in new csv numpy file
    with open('data/model_data/ensemble_final_varR.csv', 'a') as f_csv_final:
        writer2 = csv.writer(f_csv_final)
        writer2.writerow([run, accuracy, loss])
    f_csv_final.close()

        

f.close()