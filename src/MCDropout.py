# Simple CNN on FMNIST

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms


########## Load Data

# Normalize images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create datasets for training & validation
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
# improves data retrieval
training_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
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
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler to adjust the learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


############# Training the model

num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0

    # Iterate over training data in batches
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)

        # Optimization step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Step the scheduler after each epoch
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(training_loader):.4f}")


print('Training complete!')

# Save the fine-tuned model
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
        loss += criterion(predicted, labels).item() * labels.size(0)

print(f'Loss of the trained model on the test images: {loss / len(validation_loader.dataset):.4f}')
print(f'Accuracy of the trained model on the test images: {100 * correct / total:.2f}%')

'''
TODO
-----------------
- Make AL wrapper by creating loop that trains copies of the model
- Each step in this wrapper the training set is updated
- Will need multipe dropout outputs for this
'''