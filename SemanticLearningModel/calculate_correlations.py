import numpy as np
import pandas as pd
import re
import os
import glob
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from scipy import stats

""" 
a script that allows us to calculate and plot CCs between humans and networks
"""
# path strings for loading the data
nn_dir = os.getcwd() + "/relu_init_exp/averaged_inputs-outputs/relu_small_weights_16hidden_outputs_lr_"
human_dir = "C:/Users/Jirko/Desktop/Hip_Lab/analysis_scripts/data_loading/data_v2_quotes/in_out_matrix.npy"

# substrings to load each initialisation
init_strings = ["1", "0.1", "0.01", "0.001", "0.0001", "1e-05"]

def calculate_CCs(init_strings, human_dir, nn_dir):
    """_summary_

    Args:
        init_strings (_type_): _description_
        human_dir (_type_): _description_
        nn_dir (_type_): _description_

    Returns:
        _type_: _description_
    """

    # generate an empty dict of CCs for each initialisation
    cc_dict = {}
    # create an entry for our general CCs
    cc_dict["overall"] = []

    # load the human data
    human_data = np.load(human_dir)
    human_data_resh = human_data.reshape(28,8)

    for init in init_strings:
        
        # load the model data and reshape
        model_data = np.load(nn_dir + init + ".npy")
        model_data_resh = model_data.reshape(28,8)

        # calculate CC for each epoch and block and append to dict
        for epoch in range(human_data_resh.shape[1]):
            res = stats.pearsonr(human_data_resh[:,epoch], model_data_resh[:, epoch])
            if init not in cc_dict:
                cc_dict[init] = []
            cc_dict[init].append(res[0])

        human_data_flat = human_data_resh.reshape(224)
        model_data_flat = model_data_resh.reshape(224)

        res_all = stats.pearsonr(human_data_flat, model_data_flat)
        cc_dict["overall"].append(res_all[0])

    return cc_dict

cc_dict = calculate_CCs(init_strings, human_dir, nn_dir)


def plot_CCs(cc_dict, init_strings, title, save_path):
    # markers = ['o', 'h', 'x', 'd', 'p', '^']
    blocks = np.arange(len(cc_dict["1"]))
    colors = plt.cm.Reds_r(np.linspace(.25,1,len(cc_dict["1"])))

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300, facecolor='w')

    for i, init in enumerate(init_strings):
        ax.plot(blocks, cc_dict[init], color = colors[i], label = init, marker = '.')
    
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    # set the x ticks and label
    ax.set_xticks(blocks, size=3)
    ax.set_xticklabels(blocks+1)

    # make the y ticks
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    # tick params
    ax.tick_params(direction='in', length=2, width=1)

    # x axis label
    ax.set_xlabel("Block/Epoch",fontweight='bold', fontsize = 8)

    # y axis label
    ax.set_ylabel("Correlation Coefficient (Pearson's r)",fontweight='bold', fontsize = 8)

    # make the legend
    ax.legend(title = 'variance $\sigma^2$', bbox_to_anchor=(0.84,0.75))

    # make the title
    ax.set_title(title, fontweight='bold', y=1.05)
    plt.subplots_adjust(bottom=0.15, left=.15)
    # plt.show()
    plt.savefig(save_path)

plot_CCs(cc_dict, 
        init_strings,
        "Human-Relu Network CCs by Initialisation",
        os.getcwd() + "/figures/Correlations_relu_averaged.svg"
        )

def plot_global_CCs(cc_list, init_strings, title, save_path):
    blocks = np.arange(len(cc_list))

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300, facecolor='w')

    ax.plot(blocks, cc_list, marker = 'o', color='slateblue')
    
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    # set the x ticks and label
    ax.set_xticks(blocks, size=3)
    ax.set_xticklabels(init_strings)

    # make the y ticks
    # ax.set_yticks(np.arange(0.78, .92, .02), size=3)
    # ax.set_yticklabels(np.arange(0.78, .92, .02))

    for label in ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)

    # tick params
    ax.tick_params(direction='in', length=2, width=1)

    # x axis label
    ax.set_xlabel("Initialisation Variance $\mathbf{\sigma^2}$", fontweight='bold', fontsize = 8)

    # y axis label
    ax.set_ylabel("Correlation Coefficient (Pearson's r)", fontweight='bold', fontsize = 8)

    # make the title
    ax.set_title(title, fontweight='bold', y=1.05)
    plt.subplots_adjust(bottom=0.15)

    # plt.show()
    plt.savefig(save_path)

plot_global_CCs(cc_dict["overall"], 
                init_strings, 
                "Overall Human-Relu Network CCs by Initialisation",
                os.getcwd() + "/figures/Correlations_relu_overall_averaged.svg")
# sum_cc = 0
# sum_cc+= res[0]
# print(epoch)
# print(res)
