from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# task specific parameters
n_examples = 200     # n_examples should be divisible by n_classes
n_features = 6
n_classes = 4

# hyperparameters
input_size = n_classes
hidden_size = 100
output_size = n_features
n_epochs = 80
batch_size = 10
learning_rate = 0.01

# a dictionary of indices of present features in each class
class_index_dict = {"a1_b1": (0, 2), "a1_b2": (0, 3),
                    "a2_c1": (1, 4), "a2_c2": (1, 5)
                    }


class DatasetGenerator():
    """our dataset generator class"""
    def __init__(self, n_examples, n_features, n_classes, class_index_dict):
        self.n_examples = n_examples
        self.n_features = n_features
        self.n_classes = n_classes
        self.class_index_dict = class_index_dict

        # the number of training examples for each class
        self.n_per_class = int(self.n_examples / self.n_classes)

        # blank matrix for input training features, size(n_features, n_examples)
        self.features = torch.zeros(self.n_features, self.n_examples)

        # a blank vector for our class labels, size(1, n_examples)
        self.labels = torch.zeros(self.n_examples)

    def generate_dataset(self):

        # looping through our class index dict and adding 1's at indices in which features are present
        for count, (key, value) in enumerate(self.class_index_dict.items()):

            lower_col_idx = count * self.n_per_class
            higher_col_idx = lower_col_idx + self.n_per_class

            self.features[value[0], lower_col_idx: higher_col_idx] = 1
            self.features[value[1], lower_col_idx: higher_col_idx] = 1
            self.labels[lower_col_idx: higher_col_idx] = count

        self.labels = F.one_hot(self.labels.to(torch.int64), num_classes=self.n_classes)
        self.labels = self.labels.to(torch.float32)
        self.features = self.features.t()
        return self.features, self.labels


# load the x inputs and y labels into a simple pytorch Dataset, for data loading
class CustomDataset(Dataset):
    def __init__(self, input_tensors):
        assert input_tensors[0].size(0) == input_tensors[0].size(0)
        self.input_tensors = input_tensors

    def __getitem__(self, index):
        x = self.input_tensors[0][index]
        y = self.input_tensors[1][index]
        return x, y

    def __len__(self):
        return self.input_tensors[0].size(0)


# define our model class
class FullyConnected(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnected, self).__init__()
        self.fully_con1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fully_con2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fully_con1(x)
        x = self.relu(x)
        out = self.fully_con2(x)
        return out


if __name__ == "__main__":

    """
    we generate first a general and then a pytorch specific dataset, to feed it to our dataloader
    to change prediction from features to class just swap the two terms: labels, features
    """
    data_generator = DatasetGenerator(n_examples, n_features, n_classes, class_index_dict)
    features, labels = data_generator.generate_dataset()
    train_set = CustomDataset(input_tensors=(labels, features))
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True)

    """
    define the network, loss-function, and optimiser
    """
    network = FullyConnected(input_size, hidden_size, output_size)
    print(network)

    loss_func = nn.MSELoss()
    optimiser = optim.SGD(network.parameters(), lr=learning_rate)

    # lets train
    loss_history = []
    for epoch in range(n_epochs):
        correct = 0
        for i, data in enumerate(train_loader):
            x, y = data

            # first forward pass
            out = network(x)

            loss = loss_func(out, y)

            # backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # use y and x as numpy arrays for taking track of correct categorisations
            out_rounded = np.round(out.cpu().detach().numpy(), decimals=1)
            y_rouned = np.round(y.cpu().detach().numpy(), decimals=1)
            correct += np.sum(np.all(out_rounded == y_rouned, axis=1))

            if (i + 1) % (len(train_set)/batch_size) == 0:
                loss_history.append(loss.item())
                print("Epoch[{}/{}], Loss: {:.4f}, Accuracy: {} %".format(epoch+1, n_epochs,
                                                                 loss.item(),
                                                                 (100 * correct / len(train_set))))
    plt.plot(np.linspace(0, n_epochs, n_epochs), loss_history)
    plt.show()















