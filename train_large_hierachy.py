from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from SemanticLearningModel.Semantic_learning import DatasetGenerator, CustomDataset, FullyConnected


# inputs and hyperparams for dataset containing a3 and a4 as additional hierarchical nodes
n_examples = 400    # n_examples should be divisible by n_classes
n_features = 8
n_classes = 8

# hyperparameters
input_size = n_classes
hidden_size = 100
output_size = n_features
n_epochs = 80
batch_size = 10
learning_rate = 0.01


class_index_dict_large = {"a1_b1": (0, 4), "a1_b2": (0, 5),
                          "a2_c1": (1, 6), "a2_c2": (1, 7),
                          "a3_b1": (2, 4), "a3_b2": (2, 5),
                          "a4_c1": (3, 6), "a4_c2": (3, 7)
                          }


if __name__ == "__main__":

    """
    we generate first a general and then a pytorch specific dataset, to feed it to our dataloader
    first training set and then test set
    """
    # train data
    train_data_generator = DatasetGenerator(n_examples,
                                            n_features,
                                            n_classes,
                                            class_index_dict_large
                                            )

    train_features, train_labels = train_data_generator.generate_dataset()
    train_set = CustomDataset(input_tensors=(train_labels, train_features))

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True
                              )


    # test data
    test_data_generator = DatasetGenerator(int(n_examples/2),
                                           n_features,
                                           n_classes,
                                           class_index_dict_large
                                           )

    test_features, test_labels = test_data_generator.generate_dataset()
    test_set = CustomDataset(input_tensors=(test_labels, test_features))
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=True
                             )

    """
    define the network, loss-function, and optimiser
    """
    network = FullyConnected(input_size,
                             hidden_size,
                             output_size)

    print(network)

    loss_func = nn.MSELoss()
    optimiser = optim.SGD(network.parameters(), lr=learning_rate)

    # let's train
    loss_history_train = []
    loss_history_test = []

    for epoch in range(n_epochs):
        correct = 0
        running_loss_train = 0
        running_loss_test = 0

        for i, data in enumerate(train_loader):
            x_train, y_train = data

            # first forward pass
            out = network(x_train)

            # compute loss
            loss = loss_func(out, y_train)

            # backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # keep track of loss
            running_loss_train += loss.item()

            # use y and x as numpy arrays for taking track of correct categorisations
            out_rounded = np.round(out.cpu().detach().numpy(), decimals=2)
            y_rouned = np.round(y_train.cpu().detach().numpy(), decimals=2)
            correct += np.sum(np.all(out_rounded == y_rouned, axis=1))

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x_test, y_test = data

                # first forward pass
                out_test = network(x_test)
                loss_test = loss_func(out_test, y_test)
                running_loss_test += loss_test.item()

        # calculate train and test loss for epoch, append, and print

        epoch_train_loss = running_loss_train / len(train_loader)
        loss_history_train.append(epoch_train_loss)

        epoch_test_loss = running_loss_test / len(test_loader)
        loss_history_test.append(epoch_test_loss)

        print("Epoch[{}/{}], Train Loss: {:.4f}, Accuracy: {} %".format(epoch + 1, n_epochs,
                                                                  epoch_train_loss,
                                                                  (100 * correct / len(train_set))))
        print("Epoch[{}/{}],Test Loss: {:.4f}".format(epoch+1, n_epochs, epoch_test_loss))


    plt.plot(np.linspace(0, n_epochs, n_epochs), loss_history_train)
    plt.plot(np.linspace(0, n_epochs, n_epochs), loss_history_test)
    plt.show()

