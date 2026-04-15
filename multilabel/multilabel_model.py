from torch import nn

# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class MultiCNN(nn.Module):
    def __init__(self, num_notes, kernel_size=(3, 3)):
        super().__init__()

        self.kernel_size = kernel_size

        # Learn same features of the spectogram in this MultiCNN
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
            nn.Linear(128 * 16 * 21, 256),  # fix the linear output for multi label class
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_notes),
        )



    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        logits = self.note_classification(x)
        return logits
    