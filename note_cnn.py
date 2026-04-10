from torch import nn


# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CNN(nn.Module):
    def __init__(self, num_notes, kernel_size=(3, 3)):
        super().__init__()

        self.kernel_size = kernel_size

        self.conv = nn.Sequential(
            # Use a kernel size
            nn.Conv2d(1, 32, kernel_size=self.kernel_size, stride=1, padding=1), # input channel (1 for grayscale), output channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=1), # input channels -> 32 (from previous conv layer)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten = nn.Flatten()

        # Layer to classify the notes
        self.note_classification = nn.Sequential(
            nn.Linear(128 * 10 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # used to reduce overfitting
            nn.Linear(256, num_notes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        logits = self.note_classification(x)
        return logits
    

import dataset
import torch.optim as optim
import train
def train_model():
    # get data from dataloader
    train_dataset, val_dataset, test_dataset, num_classes = dataset.data_pipeline(feature_id='note')

    # Set training params
    lr = 0.001
    model = CNN(num_notes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    save_path = "note_cnn.pth"

    # train model
    print('Begin training...')
    train_losses = train.train(model, train_dataset, optimizer = optimizer, criterion = criterion, save_path = save_path, epochs = 10) # 10 is a suffiecient number of epochs

    # Evaluate training loss

    train.graph_loss(train_losses, save_figure="note_cnn_training_loss.png")


if __name__ == "__main__":
    train_model()