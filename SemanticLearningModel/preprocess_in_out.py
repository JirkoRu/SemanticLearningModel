import numpy as np
import pandas as pd
import re
import os
import glob
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import shutil

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

def sort_and_reduce(dname_nn_input, dname_nn_output, save_file, m):
    """
    sort n matrices using the sort_matrices functions, 
    remove randomisation imposed during training.
    we average m matrices.
    """
    # subdirectories of all initilisations 
    init_dir_list = os.listdir(dname_nn_input)

    for j, dir in enumerate(init_dir_list):
        # make the folder for the lr
        if os.path.exists(save_file + dir + "/"):
            shutil.rmtree(save_file + dir + "/")
        os.makedirs(save_file + dir + "/")

        for i in range(48):
            # define file strings fml
            input_str = "inputs_run_%s.npy" % (i)
            output_str = "outputs_run_%s.npy" % (i)
            # load the data
            inputs = np.load(dname_nn_input + dir + "/" + input_str)
            outputs = np.load(dname_nn_output + dir + "/" + output_str)

            # print(dname_nn_output + dir + "/" + input_str)

            # sort the arrays
            sorted_inputs, sorted_outputs = sort_matrices(inputs, outputs)

            # only include every mth tensor
            reduced_outputs = sorted_outputs[:,:,::m] 

            # change dim of outputs
            reduced_outputs = np.swapaxes(reduced_outputs,0,1)

            np.save(save_file + dir + "/" + output_str, reduced_outputs)


def sort_and_average(dname_nn_input, dname_nn_output, save_file, m):
    """
    sort n matrices using the sort_matrices functions, 
    remove randomisation imposed during training.
    We only retain every mth input-ouput matrix produced during
    training. 

    Args:
        init_strings (_type_): _description_
        m (_type_): _description_
    """

    # subdirectories of all initilisations 
    init_dir_list = os.listdir(dname_nn_input)

    for k, dir in enumerate(init_dir_list):

        # make the folder for the init (I am dumb and named it wrongly)
        if os.path.exists(save_file + dir + "/"):
            shutil.rmtree(save_file + dir + "/")
        os.makedirs(save_file + dir + "/")

        for i in range(48):
            # define file strings fml
            input_str = "inputs_run_%s.npy" % (i)
            output_str = "outputs_run_%s.npy" % (i)
            # load the data
            inputs = np.load(dname_nn_input + dir + "/" + input_str)
            outputs = np.load(dname_nn_output + dir + "/" + output_str)

            # print(dname_nn_output + dir + "/" + input_str)

            # sort the arrays
            sorted_inputs, sorted_outputs = sort_matrices(inputs, outputs)
        
            # make empty array for averages
            averaged_outputs = np.full((sorted_outputs.shape[0], 
                                        sorted_outputs.shape[1], 
                                        int(sorted_outputs.shape[2]/m)), np.nan)

            # average m tensors at a time
            for j in range(int(sorted_outputs.shape[2]/m)):
                averaged_outputs[:,:,j] = np.mean(sorted_outputs[:,:,j*m:(j+1)*m], 2)

            # change dim of outputs
            averaged_outputs = np.swapaxes(averaged_outputs,0,1)
            np.save(save_file + dir + "/" + output_str, averaged_outputs)


def average_across_runs(load_dir, save_dir):
    """_summary_

    Args:
        load_dir (_type_): _description_
        save_dir (_type_): _description_
    """

    # subdirectories of all initilisations 
    init_dir_list = os.listdir(dname_nn_input)

    # make the folder for averages across networks
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # loop over those
    for k, dir in enumerate(init_dir_list):
        # string to save file to
        save_string = dir + ".npy"

        # empty 4D array for storing
        all_array = np.full((7,4,8,48), np.nan)

        # loop over all runs, load and attach

        for i in range(48):
            output_str = "outputs_run_%s.npy" % (i)
            outputs = np.load(load_dir + dir + "/" + output_str)
            all_array[:,:,:,i] = outputs
        
        # take the mean 
        mean_array = np.mean(all_array, axis=3)
        np.save(save_dir + "/" + save_string, mean_array)


if __name__ == "__main__":

    # dir to load from
    # dname_nn_input = os.getcwd() + "/relu_sig_init_exp_48rep_2/saved_inputs-outputs/"
    # dname_nn_output = os.getcwd() + "/relu_sig_init_exp_48rep_2/saved_inputs-outputs/"
    # save_file_averaged = os.getcwd() + "/relu_sig_init_exp_48rep_2/averaged_inputs-outputs/"
    # save_file_reduced = os.getcwd() + "/relu_sig_init_exp_48rep_2/reduced_inputs-outputs_100/"

    # sort and reduce relevant files
    # sort_and_reduce(dname_nn_input, dname_nn_output, save_file_reduced , 500)
    # sort_and_average(dname_nn_input, dname_nn_output, save_file_averaged, 500)

    # list of relevant network data
    input_str_list = ["/linear_sig_init_exp_48rep_2",
                    "/relu_sig_init_exp_48rep_2"]

    for dir in input_str_list: 
        # input files
        dname_nn_input = os.getcwd() + dir + "/saved_inputs-outputs/"
        dname_nn_output = os.getcwd() + dir + "/saved_inputs-outputs/"
        # dir to save to
        save_file_averaged = os.getcwd() + dir + "/averaged_inputs-outputs/"
        save_file_reduced = os.getcwd() + dir + "/reduced_inputs-outputs_100/"
        average_across_runs(save_file_averaged, os.getcwd() + dir + "/averaged_inputs-outputs_all_runs/")
        average_across_runs(save_file_reduced, os.getcwd() + dir + "/reduced_inputs-outputs_all_runs/")