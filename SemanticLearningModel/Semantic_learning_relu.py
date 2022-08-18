from audioop import bias
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Semantic_learning import DatasetGenerator, CustomDataset
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
n_epochs = 400
batch_size = 4
learning_rate = 1/n_examples

# a dictionary of indices of present features in each class
class_index_dict = {"a1_b1": (0, 1, 3), "a1_b2": (0, 1, 4),
                    "a2_c1": (0, 2, 5), "a2_c2": (0, 2, 6)
                    }
                
# define our model class
class FullyConnected(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnected, self).__init__()

        self.fully_con1 = nn.Linear(input_size, hidden_size)
        torch.nn.init.normal_(self.fully_con1.weight, mean=0, std=0.0001/input_size)
        torch.nn.init.normal_(self.fully_con1.bias, mean=0, std=0.0001/input_size)

        self.relu = nn.ReLU()

        self.fully_con2 = nn.Linear(hidden_size, output_size)
        torch.nn.init.normal_(self.fully_con2.weight, mean=0, std=0.0001/output_size)
        torch.nn.init.normal_(self.fully_con2.bias, mean=0, std=0.0001/output_size)

    def forward(self, x):
        x = self.fully_con1(x)
        x = self.relu(x)
        out = self.fully_con2(x)
        hidden_act = x
        return hidden_act, out


if __name__ == "__main__":

    """
    we generate first a general and then a pytorch specific dataset, to feed it to our dataloader
    first training set and then test set
    """
    # train data
    train_data_generator = DatasetGenerator(n_examples,
                                            n_features,
                                            n_classes,
                                            class_index_dict
                                            )

    train_features, train_labels = train_data_generator.generate_dataset()
    train_set = CustomDataset(input_tensors=(train_labels, train_features))

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True
                              )


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

    # save the model weights
    torch.save(network.state_dict(), dname + "saved_weights/relu_small_weights_16hidden.pt")

    # little plotty plot
    plt.plot(np.linspace(0, n_epochs, n_epochs), loss_history_train, label="train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.show()
    plt.savefig(dname + "figures/loss_semantic_learning_relu_16hidden.png")