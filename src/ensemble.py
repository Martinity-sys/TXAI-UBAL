# Simple CNN on FMNIST

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchensemble import VotingClassifier


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

# Define ensemble (using GPU if available)

model = VotingClassifier(
    estimator = Net,
    n_estimators=5,
    cuda = torch.cuda.is_available(), 
)

# Set criterion and optimizer
criterion = nn.CrossEntropyLoss()
model.set_criterion(criterion)

model.set_optimizer(
    'Adam',
    lr = 0.001,
)


############# training the model

# Train the model
model.fit(train_loader=training_loader, epochs = 5, save_model = True, save_dir= "./models/")

print('Training complete!')


########### Evaluate Model

# Set the model to evaluation mode
accuracy = model.predict(validation_loader)



# print(f'Loss of the fine-tuned model on the test images: {loss / len(validation_loader.dataset):.4f}')
print(f'Accuracy of the fine-tuned model on the test images: {accuracy:.2f}%')

'''
TODO
-----------------
- Make AL wrapper by creating loop that trains copies of the model
- Each step in this wrapper the training set is updated
- Will need individual ensemble outputs for this, see how this works with ensemble package
'''