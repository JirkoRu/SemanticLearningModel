from importlib import import_module
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Semantic_learning import initialize_weights, DatasetGenerator, CustomDataset, FullyConnected
import os

dname = os.getcwd() + "/PycharmProjects/SemanticLearningModel/SemanticLearningModel/"

# task specific parameters
n_examples = 4     # n_examples should be divisible by n_classes
n_features = 7
n_classes = 4

# hyperparameters
input_size = n_classes
hidden_size = 16
output_size = n_features
n_epochs = 800
batch_size = 4
learning_rate = 1/n_examples

# a dictionary of indices of present features in each class
class_index_dict = {"a1_b1": (0, 1, 3), "a1_b2": (0, 1, 4),
                    "a2_c1": (0, 2, 5), "a2_c2": (0, 2, 6)
                    }


if __name__ == "__main__":

    """
    we generate first a general and then a pytorch specific dataset, to feed it to our dataloader
    to change prediction from features to class just swap the two terms: labels, features
    """
    # train data
    data_generator = DatasetGenerator(n_examples, n_features, n_classes, class_index_dict)
    features, labels = data_generator.generate_dataset()
    train_set = CustomDataset(input_tensors=(labels, features))
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True)
    # test data
    test_data_generator = DatasetGenerator(n_examples,
                                           n_features,
                                           n_classes,
                                           class_index_dict
                                           )

    test_features, test_labels = test_data_generator.generate_dataset()
    test_set = CustomDataset(input_tensors=(test_labels, test_features))
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False
                             )

    """
    define the network, loss-function, and optimiser
    """
    network = FullyConnected(input_size, hidden_size, output_size)
    # network.apply(initialize_weights)
    print(network)

    loss_func = nn.MSELoss()
    optimiser = optim.SGD(network.parameters(), lr=learning_rate)

    # lets train
    # make a history of losses
    loss_history_train = []

    # save the inputs and outputs in a list
    inputs  = np.full([n_classes, n_classes, n_epochs], np.nan)
    outputs = np.full([n_classes, n_features, n_epochs], np.nan)


    for epoch in range(n_epochs):
        correct = 0
        running_loss_train = 0

        for i, data in enumerate(train_loader):
            x_train, y_train = data

            # first forward pass
            hidden_act, out = network(x_train)

            # compute loss
            loss = loss_func(out, y_train)

            # backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # keep track of loss
            running_loss_train += loss.item()

            # use y and x as numpy arrays for keeping track of correct categorisations
            out_rounded = np.round(out.cpu().detach().numpy(), decimals=2)
            y_rouned = np.round(y_train.cpu().detach().numpy(), decimals=2)
            correct += np.sum(np.all(out_rounded == y_rouned, axis=1))

            # store the outputs and inputs
            inputs[:, :, epoch] = x_train.detach().numpy()
            outputs[:,:, epoch] = out.detach().numpy()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x_test, y_test = data

                # first forward pass
                hidden_act, out_test = network(x_test)
                loss_test = loss_func(out_test, y_test)

        # calculate train and test loss for epoch, append, and print

        epoch_train_loss = running_loss_train / len(train_loader)
        loss_history_train.append(epoch_train_loss)

        print("Epoch[{}/{}], Train Loss: {:.4f}, Accuracy: {} %".format(epoch + 1, n_epochs,
                                                                  epoch_train_loss,
                                                                  (100 * correct / len(train_set))))

    # save the input output matrices
    # np.save(dname + "saved_input-outputs/linear_small_weights_16hidden_inputs.npy", inputs)
    # np.save(dname + "saved_input-outputs/linear_small_weights_16hidden_outputs.npy", outputs)

    # save the model weights
    #torch.save(network.state_dict(), dname + "saved_weights/relu_small_weights_16hidden.pt")

    # little plotty plot
    plt.plot(np.linspace(0, n_epochs, n_epochs), loss_history_train, label="train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.show()
    plt.savefig(dname + "figures/loss_semantic_learning_linear_16hidden.svg")

