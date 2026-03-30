import torch
# from torch.utils.data import Dataset
# from torchvision.io import read_image
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder




class NoteDatasetOld(Dataset):
    def __init__(self, features, note_labels, duration_labels):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # (N, 1, 84, 128)
        self.note_labels = torch.tensor(note_labels, dtype=torch.long)
        # self.duration_labels = torch.tensor(duration_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.note_labels[idx] #, self.duration_labels[idx]
    


class NoteDataset(Dataset):
    def __init__(self, features, note_labels):
        self.note_labels = note_labels
        self.features = features

    def __len__(self):
        return len(self.note_labels)

    def __getitem__(self, idx):
        # get the item in tensor format to feed into the cnn
        feature = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)  # (1, 84, 128)
        label = torch.tensor(self.note_labels[idx], dtype=torch.long)  # get the note label from
        return feature, label
    

def data_pipeline(features = "./test_files/features.npy", labels = "./test_files/labels.csv", dataloader_train_path="./dataloader/train.pt", dataloader_val_path="./dataloader/val.pt", dataloader_test_path="./dataloader/test.pt", split = [.8, .2], load = True):
    # Load dataset
    features = np.load(features)                          # (N, 84, 128)
    df = pd.read_csv(labels)

    print(features.shape) 

    # Encode labels to numeric representation
    le = LabelEncoder()
    y = le.fit_transform(df['note'].values)       
    num_classes = len(le.classes_)  

    print(f"Total notes: {len(y)}")
    print(f"Unique note classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")

    if load:
        try:
            dataloader_train = torch.load(dataloader_train_path)
            dataloader_val = torch.load(dataloader_val_path)
            dataloader_test = torch.load(dataloader_test_path)
            return
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    else:

        dataset = NoteDataset(features, y)
        # Train/val split
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, split
        )
        val_dataset, test_dataset = torch.utils.data.random_split(
            val_dataset, [.5, .5]
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Save the dataloaders
        # dataloader_train = torch.save(train_loader, dataloader_train_path)
        # dataloader_val = torch.save(val_loader, dataloader_val_path)
        # dataloader_test = torch.save(test_loader, dataloader_test_path)

    return train_loader, val_loader, test_loader, num_classes
