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

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


ds = SteerDataSet("/home/morgan/rvss2023/rvss/RVSS_Need4Speed/on_laptop/augmented_data/",".jpg",transform)

print("The dataset contains %d images " % len(ds))

ds_dataloader = DataLoader(ds,batch_size=1,shuffle=True)


# Define the neural network for the classifier
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7744, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net().to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])

count_0 = 0
count_1 = 0
count_2 = 0

net.load_state_dict(torch.load('quackNet2_2.pt', map_location = torch.device('cpu')))
# torch.save(net.state_dict(), "quackNet2.pt")
net.eval()
print("Test data accuracy")
# test data
# correct = 0
# total = 0
pred = []
gt = []
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in DataLoader(test_dataset):
        # images, labels = data
        images = data["image"]
        labels = data["direction"]
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()
        pred += predicted.tolist()
        gt += labels.tolist()

pred = np.array(pred)
gt = np.array(gt)
for i in range(3):
    rel = pred[gt == i]
    acc = np.sum(rel == i)/len(rel)
    print(f'Class {i} has accuracy {100.*acc}% with {len(rel)} samples')



# print(f'Accuracy of the network on the test images: {100 * correct // total}% with {} samples')



print('Finished Training')
