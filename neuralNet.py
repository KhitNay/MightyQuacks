# Creating a custom classifier based on image imports using pyTorch
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from steerDS import SteerDataSet
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load custom data
ds = SteerDataSet("/home/khit/MightyQuacks/augRedTrack/",".jpg",transform)

# Set up parameters for splitting data
# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
validationSplit = 0.2
shuffle = True
randomSeed = 14

datasetSize = len(ds)
indices = list(range(datasetSize))
split = int(np.floor(validationSplit * datasetSize))

# Randomly shuffle data
if shuffle:
    np.random.seed(randomSeed)
    np.random.shuffle(indices)

# Split data
trainIndices, valIndices = indices[split:], indices[:split]

# Data samplers and loaders
trainSampler = SubsetRandomSampler(trainIndices)
validSampler = SubsetRandomSampler(valIndices)

trainLoader = DataLoader(ds, batch_size=1, sampler=trainSampler)
validationLoader = DataLoader(ds, batch_size=1, sampler=validSampler)

print("The dataset contains %d images " % datasetSize)

# ds_dataloader = DataLoader(dsedge_cases/" + ,batch_size=1,shuffle=True)


# Define the neural network for the classifier
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(1296, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 3)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.lstm = nn.LSTM(84, 50, 2, batch_first=True)
        self.fc3 = nn.Linear(50, 3)
        self.pos_enc = nn.Embedding(10, 50) # position encoding for 10 time steps

    def forward(self, x, pos):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1) # add a time step dimension
        pos = self.pos_enc(pos) # encode the position
        x = x + pos # add position encoding to feature vector
        x, (hn, cn) = self.lstm(x)
        x = self.fc3(x[:, -1, :]) # take the output from the final time step
        return x
    
net = Net().to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    for batch, data in enumerate(data_loader, 0):

        X = data["image"]
        y = data["direction"]

        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for batch, data in enumerate(data_loader):

            X = data["image"]
            y = data["direction"]
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

epochs = 10
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=trainLoader, 
        model=net, 
        loss_fn=criterion,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=validationLoader,
        model=net,
        loss_fn=criterion,
        accuracy_fn=accuracy_fn,
        device=device
    )


torch.save(net.state_dict(), "quackNet16.pt")