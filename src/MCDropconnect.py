# TXAI - Active Learning using Uncertainty Estimation: Monte Carlo Dropconnect
# Simple CNN on FMNIST
# Group 20: Jiri Derks and Martijn van der Meer

# imports
import copy
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchnlp.nn import WeightDropLinear


import torchvision
import torchvision.transforms as transforms

import argparse

########## Load Data

# Set Hyperparameters
argparser = argparse.ArgumentParser(description='Active Learning with Monte Carlo Dropconnect')
argparser.add_argument('--runs', type=int, default=5, help='number of runs')
argparser.add_argument('--save', type=bool, default=False, help='save model')
argparser.add_argument('--init', type=int, default=40, help='initial labeled set size')
argparser.add_argument('--acq', type=int, default=40, help='acquisition size')
argparser.add_argument('--max', type=int, default=2000, help='maximum labeled set size')
argparser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
argparser.add_argument('--t', type=int, default=25, help='number of forward passes for uncertainty quantification')
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
# Dropconnect on linear layers


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Initialise some layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.fc1 = nn.Linear(9216, 128)
        self.dropconnect1 = WeightDropLinear(9216, 128, weight_dropout=0.25)
        # self.fc2 = nn.Linear(128, 128)
        self.dropconnect2 = WeightDropLinear(128, 128, weight_dropout=0.5)
        self.fc3 = nn.Linear(128,10)

    def forward(self, x):
        # Define how the layers connect in a forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropconnect1(x)
        x = F.relu(x)
        x = self.dropconnect2(x)
        x = F.relu(x)
        x = self.fc3(x)
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

f = open("data/tst/dataMCConnect_new_varR.csv", 'w', newline='')
writer = csv.writer(f)
writer.writerow(['run', 'train_size', 'Loss', 'Accuracy'])

for run in range(N_RUNS):
    print(f'RUN {run}')
    model = Net()

    # Set the model to training mode and use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # select initial AL labeled indices list
    al_indices = torch.randint(high=len(full_training_set), size=(INIT_SIZE,))
    rem_indices = torch.tensor(range(0, len(full_training_set)))
    mask = np.ones(rem_indices.shape[0], dtype=bool)
    mask[al_indices] = False
    rem_indices = rem_indices[mask]


    # Create data loaders for our datasets; shuffle for training, not for validation
    # improves data retrieval
    curr_train = torch.utils.data.Subset(full_training_set, al_indices)
    training_loader = torch.utils.data.DataLoader(curr_train, batch_size=64, shuffle=True)

    curr_rem = torch.utils.data.Subset(full_training_set, rem_indices)
    rem_loader = torch.utils.data.DataLoader(curr_rem, batch_size=512, shuffle=False)

    
    ############# Training the model
        

    train_size = INIT_SIZE

    accuracy = []

    while(train_size <= ACQ_MAX):

        #print(f"Current Training Set Size: {train_size}")

        # reinitialize model (deepcopy does not work with the weightdrop layers)
        curr_model = Net()
        curr_model.to(device)
        optimizer = optim.Adam(curr_model.parameters(), lr=0.001)

        # Learning rate scheduler to adjust the learning rate
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        curr_model.train()
        

        # Training loop
        final_loss = 0
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0

            # Iterate over training data in batches
            for images, labels in training_loader:
                images, labels = images.to(device), labels.to(device)

                # Optimization step
                optimizer.zero_grad()
                outputs = curr_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Step the scheduler after each epoch
            # scheduler.step()

            #print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(training_loader):.4f}")

            if epoch == NUM_EPOCHS - 1:
                final_loss = running_loss/len(training_loader)

        
        # curr_model.train()
        all_preds = torch.empty((0, T), dtype=torch.long, device=device)  
        for images, labels in rem_loader:

            batch_size = images.shape[0]
            curr_preds = torch.empty((batch_size, T), dtype=torch.long, device=device)

            for t in range(T):
                images, labels = images.to(device), labels.to(device)

                outputs = curr_model(images)
                _, predicted = torch.max(outputs.data, 1)

                curr_preds[:, t] = predicted 

            all_preds = torch.cat((all_preds, curr_preds), dim=0)  # Append batch results

        # print(all_preds.shape)  
        # print("Predictions complete!")

        # Calculate Uncertainty
        uncertainty = varR(all_preds, T)

        # Calculate intermediate metrics
        correct = 0
        total = 0
        curr_model.eval()
        # Intermediate Testing loop
        with torch.no_grad():
                
            # Iterate over test data in batches
            for images, labels in training_loader:
                images, labels = images.to(device), labels.to(device)
    
                outputs = curr_model(images)
                _, predicted = torch.max(outputs.data, 1)
    
                # Save relevant metrics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total

        # Store intermediate metrics in csv
        writer.writerow([run, train_size, final_loss, accuracy])
        

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
        torch.save(model.state_dict(), './models/tst/VarR/MCDropconnect' + str(run) + '.pth')
        print('Model saved!')

    ########### Evaluate Model

    # Set the model to evaluation mode
    model.eval()

    correct = 0
    total = 0
    loss = 0

    # Testing loop
    with torch.no_grad():
        
        # Iterate over test data in batches
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Save relevant metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item() * labels.size(0)

    print(f'RUN {run}: Loss of the trained model on the test images: {loss / len(validation_loader.dataset):.4f}')
    print(f'RUN {run}: Accuracy of the trained model on the test images: {100 * correct / total:.2f}%')

f.close()

'''
TODO
-----------------
- test dropout parameters
- do final run
'''