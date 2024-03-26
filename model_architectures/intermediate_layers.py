#using outputs at intermediate layers to improve accuracy

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    name = "intermediate_layers"
    intermediate_layers = True

    def __init__(self, num_classes):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc = nn.Linear(16 * 4 * 4, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.4)                             # randomly set 40% of all neurons to weight 0 to prevent overfitting

        # Intermediate output layers
        self.output1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(864,432),
            nn.Linear(432,10)
        )

        self.output2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,128),
            nn.Linear(128,10)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        output_1 = self.output1(x)  # Output after the first convolutional layer
        output_1 = F.softmax(output_1, dim=1)

        x = F.sigmoid(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        output_2 = self.output2(x)  # Output after the second convolutional layer
        output_2 = F.softmax(output_2, dim=1)

        x = x.view(-1, 16 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x, output_1, output_2

def model_summary(model):
    print("Model Summary:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Parameters: {param.numel()}")
            total_params += param.numel()
    print(f"Total Trainable Parameters: {total_params}")


if __name__ == "__main__":
    from torchsummary import summary
    model = model(10)
    print(model.name)
    summary(model, (1, 28, 28))
