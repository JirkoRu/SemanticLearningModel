import numpy as np
import pandas as pd
import re
import os
import glob
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors

# dir to load from
dname_nn_input = os.getcwd() + "/linear_init_exp/saved_inputs-outputs/linear_small_weights_16hidden_inputs_lr_"
dname_nn_output = os.getcwd() + "/linear_init_exp/saved_inputs-outputs/linear_small_weights_16hidden_outputs_lr_"

# dir to save to
save_file = os.getcwd() + "/linear_init_exp/averaged_inputs-outputs/linear_small_weights_16hidden_outputs_lr_"

def sort_matrices(inputs, outputs):
    """ 
    function to sort my inputs takes as inputs the (4x4) inputs 
    and the (4x7) outputs to produce sorted versions of the vectors.
    """

    # empty arrays of our data
    sorted_inputs  = np.full(inputs.shape, np.nan)  
    sorted_outputs = np.full(outputs.shape, np.nan)

    # now sort by taking the order from making inputs diagonal
    for epoch in range(inputs.shape[2]):

        for i in range(inputs.shape[0]):
            sorted_inputs[i,:,epoch] = inputs[inputs[:,i, epoch] == 1, :, epoch]
            sorted_outputs[i,:, epoch] = outputs[inputs[:,i, epoch] == 1, :, epoch]
    
    return sorted_inputs, sorted_outputs

def sort_and_reduce(init_strings, m):
    """
    sort n matrices using the sort_matrices functions, 
    remove randomisation imposed during training.
    we average m matrices.
    """
    for i, init in enumerate(init_strings):
        # load the data
        inputs = np.load(dname_nn_input + init + ".npy")
        outputs = np.load(dname_nn_output + init + ".npy")

        # sort the arrays
        sorted_inputs, sorted_outputs = sort_matrices(inputs, outputs)

        # only include every mth tensor
        reduced_outputs = sorted_outputs[:,:,::m] 
        reduced_inputs = sorted_inputs[:,:,::m]

        # change dim of outputs
        reduced_outputs = np.swapaxes(reduced_outputs,0,1)
        print(reduced_outputs.shape)
        np.save(save_file + init + ".npy", reduced_outputs)


def sort_and_average(init_strings, m):
    """
    sort n matrices using the sort_matrices functions, 
    remove randomisation imposed during training.
    We only retain every mth input-ouput matrix produced during
    training. 

    Args:
        init_strings (_type_): _description_
        m (_type_): _description_
    """
    for i, init in enumerate(init_strings):
        # load the data
        inputs = np.load(dname_nn_input + init + ".npy")
        outputs = np.load(dname_nn_output + init + ".npy")

        # sort the arrays
        sorted_inputs, sorted_outputs = sort_matrices(inputs, outputs)
        
        # make empty array for averages
        averaged_outputs = np.full((sorted_outputs.shape[0], 
                                    sorted_outputs.shape[1], 
                                    int(sorted_outputs.shape[2]/m)), np.nan)

        # average m tensors at a time
        for j in range(int(sorted_outputs.shape[2]/m)):
            averaged_outputs[:,:,j] = np.mean(sorted_outputs[:,:,j*m:(j+1)*m], 2)
            print(averaged_outputs[:,:,j])

        # change dim of outputs
        averaged_outputs = np.swapaxes(averaged_outputs,0,1)
        np.save(save_file + init + ".npy", averaged_outputs)

if __name__ == "__main__":
    init_strings = ["1", "0.1", "0.01", "0.001", "0.0001", "1e-05"]
    sort_and_average(init_strings, 100)