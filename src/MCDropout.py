# Simple CNN on FMNIST

# imports
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torchvision
import torchvision.transforms as transforms


########## Load Data

# AL Paramaters
INIT_SIZE = 40
ACQ_SIZE = 40
ACQ_MAX = 2000
T = 5

# Normalize images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create datasets for training & validation
full_training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# select initial AL labeled indices list
al_indices = torch.randint(high=len(full_training_set), size=(INIT_SIZE,))
rem_indices = torch.range(0, len(full_training_set) - 1)
rem_indices = rem_indices[torch.logical_not(torch.isin(rem_indices, al_indices))]

# Create data loaders for our datasets; shuffle for training, not for validation
# improves data retrieval
curr_train = torch.utils.data.Subset(full_training_set, al_indices)
training_loader = torch.utils.data.DataLoader(curr_train, batch_size=4, shuffle=True)

curr_rem = torch.utils.data.Subset(full_training_set, rem_indices)
rem_loader = torch.utils.data.DataLoader(curr_rem, batch_size=len(curr_rem), shuffle=False)


validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=False)

# Class labels
classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


########## Define model


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

model = Net()

# Set the model to training mode and use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()


############# Training the model

def varR(forward_passes, T):

    res = []
    for i in range(len(forward_passes)):
        curr_pass = forward_passes[:, i]
        f_m = len(curr_pass[curr_pass == torch.mode(curr_pass).values])
        res.append(1 - f_m/T)
    
    return res
    

train_size = INIT_SIZE

while(train_size <= ACQ_MAX):

    print(f"Current Training Set Size: {train_size}")

    # Copy new model
    curr_model = copy.deepcopy(model)
    optimizer = optim.Adam(curr_model.parameters(), lr=0.001)

    # Learning rate scheduler to adjust the learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 1

    # Training loop
    for epoch in range(num_epochs):
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
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(training_loader):.4f}")

    # Obtain forward passes
    mc_passes = torch.tensor([])
    for _ in range(T):
        for images, labels in rem_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = curr_model(images)
            _, predicted = torch.max(outputs.data, 1)

            mc_passes = torch.cat((mc_passes, predicted), dim = 0)

    # Calculate Uncertainty
    uncertainty = varR(mc_passes, T)

    # Select n most uncertain samples and move samples to training set
    new_batch = torch.topk(uncertainty, k = ACQ_SIZE).indices
    al_indices = torch.cat((al_indices, rem_indices[new_batch]))
    mask = np.ones(rem_indices.shape[0], dtype=bool)
    mask[new_batch] = False
    rem_indices = rem_indices[mask]

    # Update data loaders
    curr_train = torch.utils.data.Subset(full_training_set, al_indices)
    training_loader = torch.utils.data.DataLoader(curr_train, batch_size=4, shuffle=True)

    curr_rem = torch.utils.data.Subset(full_training_set, rem_indices)
    rem_loader = torch.utils.data.DataLoader(curr_rem, batch_size=len(curr_rem), shuffle=False)


    # Update curr training size for while loop
    train_size += ACQ_SIZE



print('Training complete!')

# Save the final model
model = curr_model
torch.save(model.state_dict(), './models/MCDropout.pth')
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

print(f'Loss of the trained model on the test images: {loss / len(validation_loader.dataset):.4f}')
print(f'Accuracy of the trained model on the test images: {100 * correct / total:.2f}%')

'''
TODO
-----------------
- test dropout parameters
- add csv writing
- add repetitions for averaging
'''