from sys import argv 
import torch
from torch import nn
from load_dataset import test_loader
from utils import load_model, evaluate_model
import re

if __name__ == "__main__":
    if len(argv) < 3:
        print("please specify the filename of model architecture and the filename of the model weights")
        quit()
    
    print("Loading model...")
    model_module = load_model(argv[1])
    model = model_module.model()
    weights_path = argv[2]
    model.load_state_dict(torch.load("trained_models/" + weights_path))
    print(f"Succesfully loaded a {model.name} model with weights from {weights_path}.")
    criterion = nn.CrossEntropyLoss()

    avg_loss, accuracy = evaluate_model(model, criterion, test_loader)
    print(f"Average test loss:  {avg_loss:.3f}")
    print(f"Accuracy:           {accuracy:.3f}")