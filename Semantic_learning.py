from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_examples = 20000     # n_examples should be divisible by n_classes
n_features = 6
n_classes = 4

# a dictionary of indices of present features in each class
class_index_dict_classic = {"a1_b1": (0, 2), "a1_b2": (0, 3),
                            "a2_c1": (1, 4), "a2_c2": (1, 5)
                            }

# hyper parameters
input_size = n_classes
hidden_size = 80
output_size = n_features
n_epochs = 200
batch_size = 20
learning_rate = 0.0001

def generate_dataset(n_examples, n_features, n_classes, class_index_dict):

    # the number of training examples for each class
    n_per_class = int(n_examples/n_classes)

    # a blank matrix for our training data size = (features, examples)
    x = torch.zeros(n_features, n_examples)
    # a blank vector for our labels
    y = torch.zeros(n_examples)

    # looping through our class index dict and adding 1's at indices in which features are present

    for count, (key, value) in enumerate(class_index_dict.items()):
        lower_col_idx = count * n_per_class
        higher_col_idx = lower_col_idx + n_per_class
        x[value[0], lower_col_idx: higher_col_idx] = 1
        x[value[1], lower_col_idx: higher_col_idx] = 1
        y[lower_col_idx: higher_col_idx] = count

    y = F.one_hot(y.to(torch.int64), num_classes=4)
    y = y.to(torch.float32)
    x = x.t()
    labels = y
    features = x
    return features, labels


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
        return F.log_softmax(out)


# we generate the dataset, to change prediction from features to class just change the two terms x_train, y_train
features, labels = generate_dataset(n_examples, n_features, n_classes, class_index_dict_classic)
train_set = CustomDataset(input_tensors=(labels, features))
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True)



# creat instance of network class
network = FullyConnected(input_size, hidden_size, output_size)
print(network)

# define loss function and optimiser
loss_func = nn.CrossEntropyLoss()
optimiser = optim.SGD(network.parameters(), lr=learning_rate)

if __name__ == "__main__":
    # training
    loss_history = []
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            x, y = data

            # first forward pass
            out = network(x)
            loss = loss_func(out, y)

            # backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if (i + 1) % 20 == 0:
                loss_history.append(loss.item())
        print("Epoch[{}/{}], Loss: {:.4f}".format(epoch, n_epochs, loss.item()))
















