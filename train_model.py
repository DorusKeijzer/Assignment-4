from sys import argv

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib.util
from datetime import datetime
from utils import load_model, evaluate_model
from configs import BATCH_SIZE
import torch.nn.init as init

from matplotlib import pyplot as plt

# Device will determine whether to run the training on GPU or CPU.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def train(net: nn.Module, train_loader, val_loader, criterion, optimizer: torch.optim.Adam, decrease_learning_rate = False):
    print("Starting training...")
    training_loss = []
    eval_loss = []
    val_loss, val_accuracy = 0,0

    for epoch in range(15):
        print(f"Epoch {epoch+1}")
        print("=====================================")
        # if deacrease_learning_rate is set to True:
        if decrease_learning_rate and epoch > 0 and epoch % 5 == 0:
            # halves learning rate every 5th epoch for choice task #1
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                lr /= 2
                param_group['lr'] = lr 
            print(f"Halved learning rate to {lr}")

        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # if the model implements intermediate layers, this will average the loss of each layer
            # if the model has only one layer, it works as normal
            outputs = model(inputs)
            if model.intermediate_layers:
                loss = 0
                for output in outputs:
                    loss += criterion(output, labels)/len(outputs)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            if i % 100 == 0 or i == len(train_loader):  # Print when a batch is completed or it's the last batch
                avg_train_loss = train_loss / ((i + 1) * BATCH_SIZE)
                print(f"Batch: {i:>3}/{len(train_loader)}, training loss: {avg_train_loss:.4f}")

        training_loss.append(avg_train_loss)
        val_loss, val_accuracy = evaluate_model(model, criterion, val_loader)
        eval_loss.append(val_loss)
        print("—————————————————————————————————————")
        print(f'Validation Loss: {val_loss:>20.4f}\nValidation Accuracy: {val_accuracy:>16.4f}')      
    print('Finished Training\n')
    return eval_loss, training_loss, val_loss, val_accuracy 


def plotLosses(eval_loss, train_loss, final_val_loss, val_accuracy, filename, show=False):
    plt.plot(eval_loss, label='Evaluation loss', color='black', linestyle='--')
    plt.plot(train_loss, label='Training loss', color='black', linestyle=':')
    plt.plot([], [], ' ', label=f"Final validation loss {final_val_loss:.3f}")
    plt.plot([], [], ' ', label=f"Final validation accuracy {val_accuracy:.3f}")

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over time')

    # Adding legend
    plt.legend()
    plt.savefig("results/"+ filename + '.png')
    print(f"Saved plot to results/{filename}.png")

    # Displaying the plot
    if show:
        plt.show()


if __name__  == "__main__":
    if len(argv) != 2:
        raise Exception("Specify a model path")
    
    model_filename = argv[1]
    print(f"Loading {model_filename}...")
    model_module = load_model(model_filename)
    model = model_module.model().to(device)
    print(f"Succesfully loaded model {model.name} from {model_filename}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)
    
    from load_dataset import train_loader, val_loader 

    eval_loss, training_loss, val_accuracy, final_val_loss = train(model, train_loader, val_loader, criterion, optimizer)


    now = datetime.now().strftime(r'%y%m%d_%H%M%S')
    store_filepath = f"trained_models/{model.name}_{now}.pth"

    plotLosses(eval_loss, training_loss, val_accuracy, final_val_loss, f"{model.name}_{now}")

    torch.save(model.state_dict(), store_filepath)
    print(f"Saved model to {store_filepath}")