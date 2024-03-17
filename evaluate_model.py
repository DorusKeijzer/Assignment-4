from sys import argv 
import importlib
import torch
from load_dataset import testloader
from utils import load_model, evaluate_model

if __name__ == "__main__":
    if len(argv) < 3:
        print("please specify a model")

    model = load_model(argv[1])
    model_path = argv[2]
    model = model.load_state_dict(torch.load(model_path))
