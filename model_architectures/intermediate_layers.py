import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    name = "intermediate_layers"
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Intermediate output layers
        self.output1 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.output2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        output_1 = self.output1(x)  # Output after the first convolutional layer
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        output_2 = self.output2(x)  # Output after the second convolutional layer
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, output_1, output_2
