from sys import argv 
import importlib
import torch
from load_dataset import testloader

def load_model(model_filename):
    model_path = f'model_architectures.{model_filename}'
    spec = importlib.util.find_spec(model_path)
    if spec is None:
        raise ImportError(f"Model '{model_filename}' not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accuracy(model)
    # Test the LeNet model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct/ total

    print(f'Accuracy of the network on the 10000 test images: {accuracy}')

    return accuracy
if __name__ == "__main__":
    if len(argv) < 3:
        print("please specify a model")

    model = load_model(argv[1])
    model_path = argv[2]
    model = model.load_state_dict(torch.load(model_path))
