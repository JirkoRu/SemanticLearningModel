import numpy as np
import pandas as pd
import re
import os
import glob
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors


dname = os.getcwd() + "/linear_init_exp/averaged_inputs-outputs/"

fig_dname = os.getcwd() + "/PycharmProjects/SemanticLearningModel/SemanticLearningModel/"

outputs = np.load(dname + "linear_small_weights_16hidden_outputs_lr_0.0001.npy")


def sort_matrices(inputs, outputs):
    """ 
    function to sort my inputs takes as inputs the (4x4) inputs 
    and the (4x7) outputs to produce sorted versions of the vectors.
    """

    # empty arrays of our data
    sorted_inputs  = np.full(inputs.shape, np.nan)  
    sorted_outputs = np.full(outputs.shape, np.nan)
    for epoch in range(inputs.shape[2]):

        for i in range(inputs.shape[0]):
            sorted_inputs[i,:,epoch] = inputs[inputs[:,i, epoch] == 1, :, epoch]
            sorted_outputs[i,:, epoch] = outputs[inputs[:,i, epoch] == 1, :, epoch]
    
    return sorted_inputs, sorted_outputs

# sorted_inputs, sorted_outputs = sort_matrices(inputs, outputs)

# print(sorted_inputs[:,:,0])
# print(sorted_outputs[:,:,0])

# def simulate_response(outputs):
#     """ 
#     a function to set the three largest values to 1, the rest to 0
#     """

#     for epoch in range(outputs.shape[2]):
#         for cat in range(outputs.shape[0]):
#             idx = np.argpartition(outputs[ cat, :, epoch], -3)[-3:]
#             outputs[ cat, idx, epoch] = 1
#             outputs[cat, outputs[cat, : , epoch] != 1, epoch] = 0

#     return outputs

# sorted_outputs = simulate_response(sorted_outputs)

""" custom functions for plotting here"""

# custom function for the cmap
from matplotlib.pyplot import ylabel



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_inout_matrix(in_out_matrix, fig_name, fig_title, unit_str, increment):
    # Define figure
    fig, axes = plt.subplots(2, 4, figsize=(10, 8), dpi=100, facecolor='w')
    cmap = plt.get_cmap('bwr')
    new_cmap = truncate_colormap(cmap,0.5, 1)

    # Average per bloc
    for i, ax in enumerate(axes.flatten()):
        im = ax.matshow(in_out_matrix[:,:,i * increment], cmap=new_cmap, vmax = 1) 

    # remove the ticks and make tick labels
    y_tick_labels = ['Derd','Lorp','Reng','Stad', 'Blap', 'Culp', 'Wost']
    x_tick_labels = ['1', '2', '3', '4']

    for i, ax in enumerate(axes.flatten()):
        
        ax.set_title(unit_str + " " + str((i+1)*500*increment), fontweight='bold', fontsize = 10)

        # make grid
        ax.set_xticks(np.arange(-.5, len(x_tick_labels)), minor=True)
        ax.set_yticks(np.arange(-.5, len(y_tick_labels)), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

        # remove ticks
        ax.tick_params(which='both', left=False, right=False, top=False, bottom=False, pad=1)

        ax.set_xticks(range(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels, fontsize=10, ha="left", rotation_mode="anchor")

        if i % 4 == 0:
            ax.set_ylabel("Properties",fontweight='bold', fontsize = 12)
            ax.set_yticks(range(len(y_tick_labels)))
            ax.set_yticklabels(y_tick_labels, fontsize=10)
            ax.tick_params(labeltop=False, labelbottom = True)

        else:
            ax.tick_params(labelleft=False, labeltop=False, labelbottom = True)

    # add a supertitle
    fig.suptitle(fig_title, fontweight='bold', fontsize = 14)

    # add x labels
    fig.text(0.462, 0.07, 'Categories', fontweight='bold', fontsize = 12, ha='center')
    fig.text(0.462, 0.49, 'Categories', fontweight='bold', fontsize = 12, ha='center')
    
    # add a colourbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.04, 0.6])
    fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout()
    # plt.show()
    
    fig.savefig(fig_name)

def plot_n_input_puts(input_dir_list, fig_dname_average, fig_dname_reduced, vars, titles, file_names):

    # loop over the dirs
    for i, model in enumerate(input_dir_list):
        input_file_averaged = os.getcwd() + model + "/averaged_inputs-outputs_all_runs/"
        input_file_reduced = os.getcwd() + model + "/reduced_inputs-outputs_all_runs/"

        # loop over initialisations
        for j in range(len(os.listdir(input_file_averaged))):

            input_outputs = np.load(input_file_averaged + "lr_%s.npy" % (vars[j]))
            
            plot_inout_matrix(input_outputs, 
                fig_dname_average + file_names[i] % (vars[j]), 
                titles [i] % (vars[j]),
                "Epoch",
                1)

        # loop over initialisations 
        for k in range(len(os.listdir(input_file_reduced))):

            input_outputs = np.load(input_file_reduced + "lr_%s.npy" % (vars[k]))

            plot_inout_matrix(input_outputs, 
                fig_dname_reduced + file_names[i] % (vars[k]), 
                titles [i] % (vars[k]),
                "Epoch",
                1)


if __name__ == "__main__":
    vars = ["1", "0.1", "0.01", "0.001", "0.0001", "1e-05"]

    titles = ["Sigmoid-Linear Network Input-Output Correlation Matrix by Epochs ($\mathbf{\sigma^2}$ = %s)",
            "Sigmoid-Relu Network Input-Output Correlation Matrix by Epochs ($\mathbf{\sigma^2}$ = %s)"
            ]

    file_names = ["linear_sig_network_lr%s.png",
                "relu_sig_network_lr%s.png"]

    # list of relevant network data
    input_str_list = ["/linear_sig_init_exp_48rep_2",
                    "/relu_sig_init_exp_48rep_2",
                    ]

    plot_n_input_puts(input_str_list, 
                        os.getcwd() + "/figures_in_out_average_all_runs_2/png/", 
                        os.getcwd() + "/figures_in_out_reduced_all_runs_2/png/", 
                        vars, 
                        titles, 
                        file_names
                        )

    # lr = "1"
    # lr2 = "1"
    # dname = os.getcwd() + "/linear_init_exp_48rep/averaged_inputs-outputs_all_runs/"

    # fig_dname = os.getcwd() + "/figures_in_out/png_figures/"

    # outputs = np.load(dname + "lr_1.npy")

    # plot_inout_matrix(outputs, 
    #                 fig_dname + "linear_network_lr%s.png"% (lr), 
    #                 "Linear Network Input-Output Correlation Matrix by Epochs ($\mathbf{\sigma^2}$ = %s)"% (lr2),
    #                 "Epoch",
    #                 1
    #                 )