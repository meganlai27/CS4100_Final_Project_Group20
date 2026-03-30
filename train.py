import torch.nn as nn
import torch

from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, F1Score

from model import CNN

# Train the model

def train(train_dataloader: torch.utils.data.DataLoader, validation_dataloader: torch.utils.data.DataLoader, model: CNN, weights=None, epochs: int = 1, lr: float = 0.01, verbose=False, num_classes = 61):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    if weights is not None:
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    f1score = F1Score(task="multiclass", num_classes=num_classes)

    all_losses = []
    epoch_losses = []
    epoch_accuracy_scores = []
    epoch_f1_scores = []
    val_losses = []

    def train_epoch(all_losses: list[float]):
        running_loss = 0
        iterations = 0

        model.train()  # Set the model to training mode

        # for batch in dataloader:
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels= batch
            labels = labels.long()
            optimizer.zero_grad()

            y_pred = model(inputs)
            pred_labels = torch.argmax(y_pred, dim=1) + 1

            if verbose:
                print(f'Pred: {pred_labels}')
                print(f'True: {labels}')

            # loss = loss_fn(y_pred, labels-1)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            loss_item = loss.item()
            running_loss += loss_item
            all_losses.append(loss_item)
            iterations += 1
        return running_loss / iterations, all_losses

    def validate_epoch():
        model.eval()  # Set the model to evaluation mode
        val_preds = torch.LongTensor()
        val_true = torch.LongTensor()
        all_losses = []


        with torch.no_grad():
            for i, batch in tqdm(enumerate(validation_dataloader), total=len(validation_dataloader)):
                inputs, labels = batch
                labels = labels.long()
                y_pred = model(inputs)
                loss = loss_fn(y_pred, labels)
                pred_labels = torch.argmax(y_pred, dim=1) + 1

                all_losses.append(loss.item())

                val_preds = torch.cat((val_preds, pred_labels))
                val_true = torch.cat((val_true, labels))

        val_accuracy = accuracy(val_preds, val_true)
        val_f1 = f1score(val_preds, val_true)
        avg_loss = sum(all_losses)/len(validation_dataloader)

        return avg_loss, val_accuracy, val_f1

    for epoch in tqdm(range(epochs)):
        print("Epoch:", epoch + 1, "/", epochs)
        # Training
        model.train()
        avg_loss, all_losses = train_epoch(all_losses)
        epoch_losses.append(avg_loss)
        # torch.save(model.state_dict(), "note_classifier.pth")

        # Validation
        val_loss, val_accuracy, val_f1 = validate_epoch()

        val_losses.append(val_loss)
        epoch_accuracy_scores.append(val_accuracy)
        epoch_f1_scores.append(val_f1)


        print(f'Epoch {epoch + 1}/{epochs} - Avg Loss: {val_loss}, Validation Accuracy: {val_accuracy}, - Validation F1: {val_f1}')

    return val_losses, epoch_accuracy_scores, epoch_f1_scores



def graph_losses(losses, title):
    print(losses)
    plt.figure()
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title(title)
    plt.show()
def graph_accuracy(acc_scores, title):
    print(acc_scores)
    plt.figure()
    plt.plot(acc_scores)
    plt.ylabel("Accuracy Score")
    plt.xlabel("Epoch")
    plt.title(title)
    plt.show()
def graph_f1(f1_scores, title):
    print(f1_scores)
    plt.figure()
    plt.plot(f1_scores)
    plt.ylabel("F1 Score")
    plt.xlabel("Epoch")
    plt.title(title)
    plt.show()