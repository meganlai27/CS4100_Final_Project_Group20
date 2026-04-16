import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np

from multilabel_model import MultiCNN

# Evaluate the model and collect the predictions of the test set
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in tqdm(loader):
            features, labels = features.to(device), labels.to(device)
            # print(features.shape)
            x = features
            # x = features.unsqueeze(1)


            target_width = 168
            current_width = x.shape[-1]

            if current_width < target_width:
                # pad on the right
                pad_amount = target_width - current_width
                x = F.pad(x, (0, pad_amount))
            elif current_width > target_width:
                # crop input based on the target width
                x = x[:, :, :, :target_width]

            
            # Get prediction from model
            outputs = model(x)
            probs = torch.sigmoid(outputs)
            
            # get classification
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


# Get hamming accuracy to get accuracy score over individual notes
def hamming_accuracy(preds, labels):
    hamming_acc = (preds == labels).mean()

    return hamming_acc

from torchmetrics import Precision, Recall, F1Score
def get_metrics(preds, labels, num_labels = 62):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    preds_t  = torch.tensor(preds).to(device)
    labels_t = torch.tensor(labels).long().to(device)

    precision_fn = Precision(task="multilabel", num_labels=62).to(device)
    recall_fn    = Recall(task="multilabel", num_labels=62).to(device)
    f1_fn        = F1Score(task="multilabel", num_labels=62).to(device)

    precision = precision_fn(preds_t, labels_t).item()
    recall    = recall_fn(preds_t, labels_t).item()
    f1        = f1_fn(preds_t, labels_t).item()

    return precision, recall, f1

def accuracy_per_note(preds, labels):
    return (preds == labels).mean(axis=0)

# Graph accuracies:
import matplotlib.pyplot as plt
def graph_accuracy_per_note(f1):
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(f1)), f1)
    plt.title("Accuracy Score per Note")
    plt.ylabel("Accuracy")
    plt.show()



import multilabel_dataset as md
from multilabel_dataset import MultiNoteDataset
def main():
    model_path = "saved_models/multinote_classifier_chords.pth"
    test_dataset_path = "./dataloader/dataloader_test_multilabel.pt"
    
    test_dataset = md.load_dataset(test_dataset_path)

    model = MultiCNN(num_notes=62)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds, labels = evaluate_model(model, test_dataset, 'cpu')

    # Calculate accuracy metrics
    hamming_acc = hamming_accuracy(preds, labels)
    print(f'Hamming Accuracy: {hamming_acc}')

    precision, recall, f1 = get_metrics(preds, labels)
    print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')

    accuracies_per_note = accuracy_per_note(preds, labels)
    print(f'Accuracy Per Note Class: {accuracies_per_note}')

    graph_accuracy_per_note(accuracies_per_note)


if __name__ == "__main__":
    main()
