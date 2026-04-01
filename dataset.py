import torch
# from torch.utils.data import Dataset
# from torchvision.io import read_image
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

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
    
"""
Generates train, validation, and test dataloaders from the features and labels files. If dataloaders already exist, it loads from the given saved files.

Parameters:
- features: Path to the .npy file containing the features
- labels: Path to the .csv file containing the note labels
- dataloader_train_path: Path to save/load the training dataloader
- dataloader_val_path: Path to save/load the validation dataloader
- dataloader_test_path: Path to save/load the test dataloader
- split: List containing the proportions for train, validation, and test splits (ex., [<train_split>, <val_split>, <test_split>] = [0.8, 0.1, 0.1])
"""
def data_pipeline(feature_id, features = "./test_files/features.npy", labels = "./test_files/labels.csv", dataloader_train_path="./dataloader/train.pt", dataloader_val_path="./dataloader/val.pt", dataloader_test_path="./dataloader/test.pt", split = [.8, .1, .1], verbose = False):
    # Load dataset from features and files
    features = np.load(features)    # (n, 84, 128) for 84 frequency bins and 128 time steps
    df = pd.read_csv(labels)

    if verbose:
        print(f"Features shape: {features.shape}")  # (n, 84, 128)

    # Encode labels to numeric representation
    le = LabelEncoder()

    # if labels, 
    # get labels of specific feature/y label
    try: 
        labels = le.fit_transform(df[feature_id].values)
    except Exception as e:
        print(f"Error encoding labels for feature_id '{feature_id}': {e}.\nAvailable columns in labels: {df.columns.tolist()}")
        raise ValueError(f"Invalid feature_id '{feature_id}'. Please check the labels CSV file for available columns.")
    # duration_labels = le.fit_transform(df['duration'].values)       
    num_labels = len(np.unique(labels))
    # num_durations = len(duration_labels.unique())  

    print(f"Total notes classes: {num_labels}")
    # print(f"Unique duration classes: {num_durations}")
    print(f"Classes: {le.classes_}")

    try:
        train_dataset = torch.load(dataloader_train_path)
        val_dataset = torch.load(dataloader_val_path)
        test_dataset = torch.load(dataloader_test_path)
        
        print(f"Loaded existing dataloaders from: {dataloader_train_path}, {dataloader_val_path}, {dataloader_test_path}")
    except:
        print(f"Creating a new dataloader to: {dataloader_train_path}, {dataloader_val_path}, {dataloader_test_path}")
        dataset = NoteDataset(features, labels)  # Use note labels for training the note model

        # Train/val split
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, split
        )

        # Save the dataloaders --> Too large to load
        # torch.save(train_dataset, dataloader_train_path)
        # torch.save(val_dataset, dataloader_val_path)
        # torch.save(test_dataset, dataloader_test_path)

    # create loaders
    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=32)
    dataloader_test = DataLoader(test_dataset, batch_size=32)

    return dataloader_train, dataloader_val, dataloader_test, num_labels


# Will be on final, finding the derivative of the loss function (should be able to get through most of lecture)