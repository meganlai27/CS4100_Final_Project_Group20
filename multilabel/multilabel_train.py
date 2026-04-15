from torch import nn
import torch
from tqdm import tqdm
from multilabel_model import MultiCNN


# First iteration of training using Adam optimizer
def train(train_dataset, model = None, num_notes=62, save_path = "multilabel_model.pth"):
    train_losses = [] # keep train of loss

    if not model:
        model = MultiCNN(num_notes=num_notes)
    
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    criterion = nn.BCEWithLogitsLoss() 

    # Used to get the binary cross entropy loss for each not category
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    correct = 0
    total = 0

    print("Starting training loop...")

    for epoch in range(2):  # loop over the dataset multiple times
        model.train()

        running_loss = 0.0
        for i, data in tqdm(enumerate(train_dataset, 0), total=len(train_dataset)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.float() 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float() # get all the labels that are classified as 1
            correct += (preds == labels).sum().item()
            total += labels.numel()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    
                avg_loss = running_loss / 50
                train_losses.append(avg_loss)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), save_path)

    print(f'Training accuracy: {correct / total:.4f}')
