import importlib
import torch

def load_model(model_filename):
    model_path = f'model_architectures.{model_filename}'
    spec = importlib.util.find_spec(model_path)
    if spec is None:
        raise ImportError(f"Model '{model_filename}' not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def evaluate_model(model, criterion, data_loader):
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            if model.intermediate_layers: # in case of multiple layers, average the loss over each layer
                for output in outputs:
                    loss += criterion(output, labels).item() * inputs.size(0)/len(outputs)
                # only the final output is used for prediction accuracy
                _, predicted = torch.max(outputs[0], 1)
            else:
                loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = loss / len(data_loader.dataset)
    accuracy = correct / total

    return avg_loss, accuracy
