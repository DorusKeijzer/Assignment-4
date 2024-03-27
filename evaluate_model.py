from sys import argv 
import torch
from torch import nn
from load_dataset import test_loader
from utils import load_model, evaluate_model
import re
from copy import deepcopy
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

if __name__ == "__main__":
    if len(argv) < 3:
        print("please specify the filename of model architecture and the filename of the model weights")
        quit()
    
    print("Loading model...")
    model_module = load_model(argv[1])
    model = model_module.model(10)
    weights_path = argv[2]
    model.load_state_dict(torch.load("trained_models/" + weights_path))
    print(f"Succesfully loaded a {model.name} model with weights from {weights_path}.")
    criterion = nn.CrossEntropyLoss()

    test_loader_copy = deepcopy(test_loader)
    model.eval()
    avg_loss, accuracy = evaluate_model(model, criterion, test_loader)
    print(f"Average test loss:  {avg_loss:.3f}")
    print(f"Accuracy:           {accuracy:.3f}")

    predictions = []
    truelabels = []
    with torch.no_grad():
        for inputs, labels in test_loader_copy:
            outputs = torch.tensor(model(inputs).numpy())

            _, predicted = torch.max(outputs, 1)
        
            predictions.extend(predicted)
            truelabels.extend(labels)


    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Build confusion matrix
    cf_matrix = confusion_matrix(truelabels, predictions)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("results/cf_matrix/"+weights_path+'.png')


    