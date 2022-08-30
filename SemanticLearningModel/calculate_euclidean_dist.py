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
# nn_dir = os.getcwd() + "/linear_init_exp_48rep/averaged_inputs-outputs_all_runs/lr_"
nn_dir = os.getcwd() + "/linear_sig_init_exp_48rep_2/averaged_inputs-outputs_all_runs/lr_"
# nn_dir = os.getcwd() + "/relu_init_exp_48rep/averaged_inputs-outputs_all_runs/lr_"
# nn_dir = os.getcwd() + "/relu_sig_init_exp_48rep_2/averaged_inputs-outputs_all_runs/lr_"

human_dir = "C:/Users/Jirko/Desktop/Hip_Lab/analysis_scripts/data_loading/data_v2_quotes/in_out_matrix.npy"

# substrings to load each initialisation
init_strings = ["1", "0.1", "0.01", "0.001", "0.0001", "1e-05"]

def calculate_Euclidean_dist(init_strings, human_dir, nn_dir):
    """_summary_

    Args:
        init_strings (_type_): _description_
        human_dir (_type_): _description_
        nn_dir (_type_): _description_

    Returns:
        _type_: _description_
    """

    # generate an empty dict of CCs for each initialisation
    eu_dict = {}
    # create an entry for our general CCs
    eu_dict["overall"] = []

    # load the human data
    human_data = np.load(human_dir)
    human_data_resh = human_data.reshape(28,8)

    for init in init_strings:
        
        # load the model data and reshape
        model_data = np.load(nn_dir + init + ".npy")
        model_data_resh = model_data.reshape(28,8)

        # calculate CC for each epoch and block and append to dict
        for epoch in range(human_data_resh.shape[1]):
            res = np.linalg.norm(human_data_resh[:,epoch] - model_data_resh[:, epoch])
            if init not in eu_dict:
                eu_dict[init] = []
            eu_dict[init].append(res)

        human_data_flat = human_data_resh.reshape(224)
        model_data_flat = model_data_resh.reshape(224)

        res_all = np.linalg.norm(human_data_flat - model_data_flat)
        eu_dict["overall"].append(res_all)

    return eu_dict

eu_dict = calculate_Euclidean_dist(init_strings, human_dir, nn_dir)


def plot_EUs(eu_dict, init_strings, title, save_path):
    # markers = ['o', 'h', 'x', 'd', 'p', '^']
    blocks = np.arange(len(eu_dict["1"]))
    colors = plt.cm.Reds_r(np.linspace(.25,1,len(eu_dict["1"])))

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.9), dpi=300, facecolor='w')

    for i, init in enumerate(init_strings):
        ax.plot(blocks, eu_dict[init], color = colors[i], label = init, marker = '.', linewidth=2.5)
    
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
    ax.set_xlabel("Block/Epoch", fontsize = 10)

    # y axis label
    ax.set_ylabel("Euclidean Distance", fontsize = 10)

    # make the legend
    ax.legend(title = 'variance $\sigma^2$', bbox_to_anchor=(0.95,0.65))

    # make the title
    ax.set_title(title, fontweight='bold', y=1.05)
    # plt.subplots_adjust(bottom=0.15, left=.15, right=.15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

plot_EUs(eu_dict, 
        init_strings,
        "Human-Linear(Sigmoid) Network Euclidean Distance",
        os.getcwd() + "/figures_euclidean_dist_2/euclidean_linear_sig_averaged_2.png"
        )

# "Human-Linear(Sigmoid) Network CCs by Initialisation"
# "/figures_correlations/Correlations_linear_sig_averaged.svg"

def plot_global_EUs(eu_list, init_strings, title, save_path):
    blocks = np.arange(len(eu_list))

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.9), dpi=300, facecolor='w')

    ax.plot(blocks, eu_list, marker = 'o', color='slateblue', linewidth=2.5)
    
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
    ax.set_xlabel("Initialisation Variance $\sigma^2$", fontsize = 10)

    # y axis label
    ax.set_ylabel("Euclidean Distance", fontsize = 10)

    # make the title
    ax.set_title(title, fontweight='bold', y=1.05)
    
    plt.subplots_adjust(bottom=0.15, right=.9)

    plt.savefig(save_path)
    plt.show()

plot_global_EUs(eu_dict["overall"], 
                init_strings, 
                "Overall Human-Linear(Sigmoid) Network Euclidean Distance",
                os.getcwd() + "/figures_euclidean_dist_2/euclidean_linear_sig_overall_averaged_2.png")

# "Overall Human-Linear(Sigmoid) Network CCs by Initialisation",
# os.getcwd() + "/figures_correlations/Correlations_linear_sig_overall_averaged.svg")