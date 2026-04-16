from torch import nn
import torch
from tqdm import tqdm
from multilabel_model import MultiCNN
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score

# Compute the weight of positives and negatives per chord segments
# This was added to weight the true positives more then true negatives during training
def compute_pos_weight(dataloader, device):
    total_pos = 0
    total_neg = 0

    for _, labels in dataloader:
        labels = labels.to(device)
        total_pos += (labels == 1).sum(dim=0)
        total_neg += (labels == 0).sum(dim=0)

    return total_neg / (total_pos + 1e-8)


def train(train_dataset, val_dataset, model=None, num_notes=62,
          save_path="multilabel_model.pth", epochs=5):

    train_losses, val_losses, val_f1_scores = [], [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = MultiCNN(num_notes=num_notes)
    model = model.to(device)

    # Compute loss 
    pos_weight = compute_pos_weight(train_dataset, device) # added to take into consideration the true positives
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Get metrics for training and validation
    def create_metrics():
        return (
            MultilabelPrecision(num_labels=num_notes, threshold=0.5, average="micro").to(device),
            MultilabelRecall(num_labels=num_notes, threshold=0.5, average="micro").to(device),
            MultilabelF1Score(num_labels=num_notes, threshold=0.5, average="micro").to(device),
        )

    def train_epoch():
        model.train()

        precision_metric, recall_metric, f1_metric = create_metrics()

        running_loss = 0.0

        for inputs, labels in tqdm(train_dataset, total=len(train_dataset)):
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(outputs)

            # update metrics
            precision_metric.update(probs, labels.int())
            recall_metric.update(probs, labels.int())
            f1_metric.update(probs, labels.int())

            running_loss += loss.item()

        # compute per training epoch
        precision = precision_metric.compute()
        recall = recall_metric.compute()
        f1 = f1_metric.compute()

        return running_loss / len(train_dataset), precision, recall, f1

    def validate_epoch():
        model.eval()

        precision_metric, recall_metric, f1_metric = create_metrics()

        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(val_dataset, total=len(val_dataset)):
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                probs = torch.sigmoid(outputs)

                precision_metric.update(probs, labels.int())
                recall_metric.update(probs, labels.int())
                f1_metric.update(probs, labels.int())

                running_loss += loss.item()

        precision = precision_metric.compute()
        recall = recall_metric.compute()
        f1 = f1_metric.compute()

        return running_loss / len(val_dataset), precision, recall, f1

    print("Starting training loop...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_prec, train_rec, train_f1 = train_epoch()
        val_loss, val_prec, val_rec, val_f1 = validate_epoch()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1.item())

        print(f"  Train Loss: {train_loss:.4f} | Precision: {train_prec:.4f} Recall: {train_rec:.4f} F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Precision: {val_prec:.4f} Recall: {val_rec:.4f} F1: {val_f1:.4f}")

    torch.save(model.state_dict(), save_path)

    return train_losses, val_losses, val_f1_scores

import matplotlib.pyplot as plt
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
    plt.plot(val_accuracies, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('Validation F1 Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_figure)

import multilabel_dataset as md

def main():
    train_dataset, val_dataset, test_dataset = md.data_pipeline(
        features_path="features_multilabel_chords.npy",
        labels_path='labels_multilabel_chords.csv'
    )

    model = MultiCNN(num_notes=62)
    save_path = "multinote_classifier_chords.pth"

    print('Begin training...')
    train_losses, val_losses, val_f1_scores = train(
        train_dataset, val_dataset, model, save_path=save_path, epochs=5
    )

    graph_losses(train_losses, val_losses, save_figure="multilabel_training_loss.png")
    graph_accuracy(val_f1_scores, save_figure="multilabel_val_f1.png")


if __name__ == "__main__":
    main()
