import torch.nn as nn
import torch

from tqdm.autonotebook import tqdm

# Train the model

def train(model, train_dataset: torch.utils.data.DataLoader, optimizer = None, criterion = nn.CrossEntropyLoss(), save_path = 'model.pth', weights=None, epochs: int = 1, lr: float = 0.01, verbose=False, num_classes = 61):
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    correct = 0
    total = 0

    train_losses = []

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        model.train()

        running_loss = 0.0
        print(f'Running Epoch {epoch + 1}...')
        for i, data in tqdm(enumerate(train_dataset, 0), total=len(train_dataset)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.long() 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    
                avg_loss = running_loss / 100
                train_losses.append(avg_loss)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    accuracy = correct / total

    print(f'Training accuracy: {accuracy:.4f}')

    return train_losses, accuracy

import matplotlib.pyplot as plt
def graph_loss(all_losses, save_figure):
    plt.figure(figsize=(8, 4))
    plt.plot(all_losses)
    plt.xlabel('Steps (every 100 batches)')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.savefig(save_figure)