from matplotlib import pyplot as plt
val=[0.7402, 0.794, 0.813, 0.8202, 0.8292, 0.8184, 0.8332, 0.8444, 0.8312, 0.854, 0.8562, 0.854, 0.852, 0.8616, 0.8596]
train=[0.7376, 0.7995, 0.8248, 0.8292, 0.8405, 0.8314, 0.852, 0.8604, 0.8482, 0.8707, 0.8702, 0.874, 0.8699, 0.8818, 0.8782]

def plot(val, train, filename):
    plt.plot(val, label='Evaluation Accuracy', color='black', linestyle='--')
    plt.plot(train, label='Training Accuracy', color='black', linestyle=':')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over time')

    # Adding legend
    plt.legend()
    plt.savefig("results/accuracies/"+ filename + '.png')
    print(f"Saved plot to results/{filename}.png")

    # Displaying the plot
    plt.show()

plot(val,
           train,
           "intermediate layers")
