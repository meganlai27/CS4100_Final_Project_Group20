from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import dataset
import torch
import torch.optim as optim
from tqdm import tqdm


# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CNN(nn.Module):
    def __init__(self, num_notes, kernel_size=(3, 3)):
        super().__init__()
        self.num_notes = num_notes

        self.conv1 = nn.Conv2d(1, 32, 5) # input channels -> 1 (grayscale), output channels, kernel size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5) # input channels -> 32 (from previous conv layer)
        self.fc1 = nn.Linear(64 * 18 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_notes)



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
'''
model = CNN(num_notes=61) # todo: get this number from the dataset
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# get data from dataloader
train_dataset, val_dataset, test_dataset = dataset.data_pipeline(load=True)


train_losses = [] # keep train of loss

print("Starting training loop...")

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in tqdm(enumerate(train_dataset, 0), total=len(train_dataset)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            avg_loss = running_loss / 2000
            train_losses.append(avg_loss)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Plot loss
import matplotlib.pyplot as plt
# Plot after training
plt.figure(figsize=(8, 4))
plt.plot(train_losses)
plt.xlabel('Steps (every 2000 batches)')
plt.ylabel('Loss')
plt.title('Training loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''