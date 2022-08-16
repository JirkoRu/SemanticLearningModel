from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Semantic_learning import DatasetGenerator, CustomDataset, FullyConnected
from train_large_hierachy import class_index_dict_large


# values for dataset containing a3 and a4 as additional higher nodes

n_examples = 8  # n_examples should be divisible by n_classes
n_features = 8
n_classes = 8

# hyperparameters
input_size = n_classes
hidden_size = 100
output_size = n_features
batch_size = 8


def make__plot_rsm(represent, save_to_path):
    """ a function to calculate, plot and save the RSM for a given in put tensor"""
    if torch.is_tensor(represent):
        rsm = torch.matmul(represent, represent.t()).numpy()
    else:
        rsm = np.dot(represent, represent)
    plt.imshow(rsm, cmap="bwr")
    plt.savefig(save_to_path)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # lets test the model and create an RSM, yay!
    # test data
    test_data_generator = DatasetGenerator(n_examples,
                                           n_features,
                                           n_classes,
                                           class_index_dict_large
                                           )

    test_features, test_labels = test_data_generator.generate_dataset()

    test_set = CustomDataset(input_tensors=(test_labels, test_features))

    # load data, note that shuffle is false!
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False
                             )

    # initalise the network
    network = FullyConnected(input_size,
                             hidden_size,
                             output_size
                             )

    # load weights
    network.load_state_dict(torch.load(("C:/Users/Jirko/PycharmProjects/SemanticLearningModel/"
                                        "SemanticLearningModel/saved_weights/linear_small_weights_100hidden.pt")))

    # make predictions
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x_test, y_test = data
            hidden_act, out_test = network(x_test)
            print(hidden_act.size())

    make__plot_rsm(hidden_act, "figures/RSM_linear_hidden_activations.png")




