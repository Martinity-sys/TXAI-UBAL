# Simple CNN on FMNIST

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

########## Load Data

# Normalize images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create datasets for training & validation
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

#split dataset into "labeled" and "unlabeled" data for active learning
labeled_data, unlabeled_data = torch.utils.data.random_split(training_set, [2000, 58000])

labeled_data_loader = torch.utils.data.DataLoader(labeled_data, batch_size=16, shuffle=True)
unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_data, batch_size=16, shuffle=False)

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
model.fit(train_loader=labeled_data_loader, epochs = 5, save_model = True, save_dir= "./models/")

print('Training complete!')


########### Evaluate Model

# Set the model to evaluation mode
accuracy = model.evaluate(unlabeled_data_loader)

print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

preds_full_set = np.zeros((5, len(unlabeled_data_loader.dataset), 10))
curr_idx = 0

for i, (images, labels) in enumerate(unlabeled_data_loader):
    print(f'Batch {i+1}/{len(unlabeled_data_loader)}')
    all_preds = np.zeros((5, 16, 10))

    for idx, net in enumerate(model.estimators_):

        tmp_single_model = net
        tmp_single_model.eval()
        
        for idx_img, image in enumerate(images):

            single_input = image.to("cuda") 
            single_out = tmp_single_model.forward(single_input.unsqueeze(0))
            
            #apply softmax
            single_out = F.softmax(single_out, dim=1)
            all_preds[idx][idx_img] = single_out.cpu().detach().numpy()
            preds_full_set[idx][curr_idx] = single_out.cpu().detach().numpy()

            if idx == model.n_estimators - 1:
                curr_idx += 1
        

    # #average the predictions
    # avg_preds = np.mean(all_preds, axis=0)
    # avg_preds = np.argmax(avg_preds, axis=1)
    # print(f'Predictions: {avg_preds}')

    # #full model predictions
    # outputs = model.forward(images.to("cuda"))
    # outputs = F.softmax(outputs, dim=1)
    # outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
    # print(f'Full model predictions: {outputs}')

#most uncertain predictions in preds_full_set
uncertainty = np.zeros(len(unlabeled_data_loader.dataset))
for idx, preds in enumerate(preds_full_set):
    for idx_img, pred in enumerate(preds):
        uncertainty[idx_img] = np.var(pred, axis=0)

#find the most uncertain predictions
most_uncertain = np.argsort(uncertainty, axis=0)
most_uncertain = most_uncertain[:1000]
print(f'Most uncertain predictions: {most_uncertain}')

#add the most uncertain predictions to the labeled dataset
labeled_data.dataset = torch.utils.data.ConcatDataset([labeled_data.dataset, torch.utils.data.Subset(unlabeled_data.dataset, most_uncertain)])
unlabeled_data.dataset = torch.utils.data.Subset(unlabeled_data.dataset, np.delete(np.arange(len(unlabeled_data.dataset)), most_uncertain))
labeled_data_loader = torch.utils.data.DataLoader(labeled_data, batch_size=16, shuffle=True)
unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_data, batch_size=16, shuffle=False)

# Train the model
model.fit(train_loader=labeled_data_loader, epochs = 1, save_model = True, save_dir= "./models/")
print('Training complete!')

# Evaluate the model
accuracy = model.evaluate(validation_loader)
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')




# print(f'Loss of the fine-tuned model on the test images: {loss / len(validation_loader.dataset):.4f}')
print(f'Accuracy of the fine-tuned model on the test images: {accuracy:.2f}%')

'''
TODO
-----------------
- Make AL wrapper by creating loop that trains copies of the model
- Each step in this wrapper the training set is updated
- Will need individual ensemble outputs for this, see how this works with ensemble package
'''