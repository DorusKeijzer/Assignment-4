from sys import argv

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib.util
from datetime import datetime


def train(net: nn.Module, trainloader, criterion, optimizer):
    for epoch in range(10):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

def load_model(model_filename):
    model_path = f'model_architectures.{model_filename}'
    spec = importlib.util.find_spec(model_path)
    if spec is None:
        raise ImportError(f"Model '{model_filename}' not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__  == "__main__":
    if len(argv) != 2:
        raise Exception("Specify a model path")

    model_filename = argv[1]
    model_module = load_model(model_filename)
    model = model_module.model()  # Assuming your model class is named Model1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.adam(model.parameters(), lr= 0.001)
    
    from load_dataset import trainloader as trainloader

    train(model, trainloader, criterion, optimizer)

    now = datetime.now().strftime(r'%y%m%d_%H%M%S')

    torch.save(model.state_dict(), f"lenet_test{now}")
