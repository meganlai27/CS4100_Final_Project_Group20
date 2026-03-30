from torch import nn
from torchvision import transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(28*28, 512),
        #     nn.ReLU(),
        #     # TODO: Create another Linear layer which takes the previous layer's outputs and produces 512 neurons.
        #     # TODO: Follow this up with a ReLU layer.
        #     # TODO: Add an output Linear layer. Think about how many output classes you have. How many output neurons would you need?
        #     nn.Linear(512, "YOUR OUTPUT NEURONS HERE"),
        # )

    def forward(self, x):
        # TODO: Flatten your input, x and then pass it through the linear_relu_stack
        x = "YOUR CODE HERE"
        logits = "YOUR CODE HERE"
        return logits