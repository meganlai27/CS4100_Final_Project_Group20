import torch.nn as nn
import torch
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

# Model training for note classification model
def train(model, train_dataset, val_dataset, optimizer=None, criterion=nn.CrossEntropyLoss(), 
          save_path='model.pth', epochs=1, lr=0.01, num_classes=61):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device) if hasattr(criterion, 'to') else criterion

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_accuracies = [], [], []

    def train_epoch():
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in tqdm(enumerate(train_dataset), total=len(train_dataset)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_dataset)
        accuracy = correct / total
        return avg_loss, accuracy

    # Evaluate the accuracies in the validation dataset during training
    def validate_epoch():
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, data in tqdm(enumerate(val_dataset), total=len(val_dataset)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.long().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        avg_loss = running_loss / len(val_dataset)
        accuracy = correct / total
        return avg_loss, accuracy

    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch + 1}/{epochs}')

        train_loss, train_acc = train_epoch()
        val_loss, val_acc = validate_epoch()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}')

    print('Finished Training')
    torch.save(model.state_dict(), save_path)

    return train_losses, val_losses, val_accuracies

# Graph and evaluate training loss & accuracy

def graph_losses(train_losses, val_losses, save_figure):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_figure)


def graph_accuracy(val_accuracies, save_figure):
    plt.figure(figsize=(8, 4))
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_figure)