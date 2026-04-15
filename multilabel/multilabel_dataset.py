from torch.utils.data import DataLoader, Dataset
import torch
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

'''
Contains the dataset pipeline to build a pytorch dataset used for training a multilabel classifier
'''


class MultiNoteDataset(Dataset):
    def __init__(self, features, note_labels, num_classes):
        self.note_labels = note_labels
        self.features = features
        self.num_classes = num_classes

    def __len__(self):
        return len(self.note_labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.note_labels[idx], dtype=torch.float32) # use float value for label array
        return feature, label
    
'''
Get all dataloaders for training
'''
def get_training_dataloaders(features, labels, split = [0.8, 0.1, 0.1], feature_id='labels', dataloader_train_path = "multilabel_train_dataloader", dataloader_val_path = "multilabel_val_dataloader", dataloader_test_path = "multilabel_test_dataloader"):
    # Try to parse all the data and get all possible labels

    try:
        features['parsed_labels'] = features[feature_id].apply(ast.literal_eval)
        mlb = MultiLabelBinarizer()
        # labels = le.fit_transform(df[feature_id].values)
        labels = mlb.fit_transform(features['parsed_labels'].values)
        print(mlb.classes_)        # all 61 possible notes
        print(labels.shape)        # (n_frames, 61)
        print(labels[0])

        labels = torch.tensor(labels, dtype=torch.float32)
    except Exception as e:
        print(f"Error encoding labels for feature_id '{feature_id}': {e}.\nAvailable columns in labels: {features.columns.tolist()}")
        raise ValueError(f"Invalid feature_id '{feature_id}'. Please check the labels CSV file for available columns.")
    

    # Get all the dataloaders with processed features and labels

    print(f"Creating a new dataloader to: {dataloader_train_path}, {dataloader_val_path}, {dataloader_test_path}")
    dataset = MultiNoteDataset(features, labels, num_classes=61)

    # Train/val split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, split
    )

    # create dataloaders
    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=32)
    dataloader_test = DataLoader(test_dataset, batch_size=32)

    return dataloader_train, dataloader_val, dataloader_test


def data_pipeline(features_path, labels_path, split = [0.8, 0.1, 0.1]):
    features = np.load(features_path)    
    labels = pd.read_csv(labels_path)

    return get_training_dataloaders(features, labels, split)



