import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class model(nn.Module):
    name = "intermediate_layers"
    intermediate_layers = True
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # output 3 (final output) is taken here

        # Intermediate output layers
        self.output1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(864,10)
        )
        # Intermediate output layers
        self.output2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,10)
        )
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        output_1 = self.output1(x)  # Output after the first convolutional layer

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        output_2 = self.output2(x)  # Output after the second convolutional layer

        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, output_1, output_2

if __name__=="__main__": 
    nn = model()

    # Print the summary
    input = torch.rand(64,1,28,28)
    nn.forward(input)