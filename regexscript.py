from sys import argv
import re
val = []
train = []
with open(argv[1], "r") as terminaloutput:
    for line in terminaloutput:
        if(valtext:= re.match(r"Validation Accuracy: +(\d\.\d+)",line)):
            val.append(float(valtext.group(1)))
        if(traintext:= re.match(r"Training Loss: +\d\.\d+\\Training Accuracy: +(\d\.\d+)",line)):
            train.append(float(traintext.group(1)))

print(f'{val=}')
print(f'{train=}')