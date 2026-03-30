import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class RhythmDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)


class RhythmCNN(nn.Module):
    def __init__(self, num_classes):
        super(RhythmCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 21 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(features)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            out = model(features)
            loss = criterion(out, labels)

            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)

    return total_loss / len(loader), correct / total


def main():
    # Load dataset
    features = np.load('features.npy')
    df = pd.read_csv('labels.csv')

    # Encode duration labels as categories
    le = LabelEncoder()
    duration_labels = le.fit_transform(df['duration'].values)

    print(f"Total notes: {len(duration_labels)}")
    print(f"Unique duration classes: {len(le.classes_)}")
    print(f"Classes: {le.classes_}")

    # Train/val split
    X_train, X_val, d_train, d_val = train_test_split(
        features, duration_labels, test_size=0.2, random_state=42
    )

    train_dataset = RhythmDataset(X_train, d_train)
    val_dataset = RhythmDataset(X_val, d_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = RhythmCNN(num_classes=len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    EPOCHS = 20
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), 'rhythm_classifier.pth')
    np.save('rhythm_label_encoder_classes.npy', le.classes_)
    print("Rhythm model saved!")

if __name__ == "__main__":
    main()