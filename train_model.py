from sys import argv

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib.util
from datetime import datetime


def train(net: nn.Module, trainloader, criterion, optimizer: torch.optim.Adam, decrease_learning_rate = False):
    print("Starting training...")
    training_loss = []
    eval_loss = []
    for epoch in range(15):
        print(f"Epoch: {epoch}")
        # if deacrease_learning_rate is set to True:
        if decrease_learning_rate and epoch > 0 and epoch % 5 == 0:
            # halves learning rate every 5th epoch for choice task #1
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                lr /= 2
                param_group['lr'] = lr 
            print(f"Halved learning rate to {lr}")

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
        training_loss.append(running_loss)

    print('Finished Training')
    return eval_loss, training_loss 

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
    print(f"Loading {model_filename}...")
    model_module = load_model(model_filename)
    model = model_module.model()  
    print(f"Succesfully loaded model {model.name} from {model_filename}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)
    
    from load_dataset import trainloader as trainloader

    train(model, trainloader, criterion, optimizer)

    now = datetime.now().strftime(r'%y%m%d_%H%M%S')
    store_filepath = f"trained_models/{model.name}_{now}.pth"

    torch.save(model.state_dict(), store_filepath)
    print(f"Saved model to {store_filepath}")