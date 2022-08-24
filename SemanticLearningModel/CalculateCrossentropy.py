from msilib import PID_TITLE
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
# from sklearn.preprocessing import normalize

""" 
a script that allows us to calculate and plot CCs between humans and networks
"""

# path strings for loading the data
nn_dir = os.getcwd() + "/linear_init_exp_48rep/averaged_inputs-outputs_all_runs/lr_"
# nn_dir = os.getcwd() + "/linear_sig_init_exp_48rep/averaged_inputs-outputs_all_runs/lr_"
# nn_dir = os.getcwd() + "/relu_init_exp_48rep/averaged_inputs-outputs_all_runs/lr_"
# nn_dir = os.getcwd() + "/relu_sig_init_exp_48rep/averaged_inputs-outputs_all_runs/lr_"

human_dir = "C:/Users/Jirko/Desktop/Hip_Lab/analysis_scripts/data_loading/data_v2_quotes/in_out_matrix.npy"

# substrings to load each initialisation
init_strings = ["1", "0.1", "0.01", "0.001", "0.0001", "1e-05"]


def cross_entropy(a, b, epsilon=1e-9):
    """A function to calcuate the crosentropy
    Args:
        a (1D np array): Distribution a 
        b (1D np array): Distribution b
        epsilon (_type_, optional): constant for normalisation. Defaults to 1e-12.

    Returns:
        float: the crossentropy
    """
    # a = np.clip(a, 1e-12, 1. - 1e-12)
    b = np.clip(b, 1e-12, 1. - 1e-12)
    # b = b + abs(np.min(b))
    # b = np.exp(b)
    a = a/np.sum(a)
    b = b/np.sum(b)
    ce = -np.dot(a, np.log(b+epsilon))
    return ce


def calculate_CEs(init_strings, human_dir, nn_dir):
    """a function to calculate the CE

    Args:
        init_strings (_type_): _description_
        human_dir (_type_): _description_
        nn_dir (_type_): _description_
    """
    # generate an empty dict of CCs for each initialisation
    ce_dict = {}
    # create an entry for our general CCs
    ce_dict["overall"] = []

    # load the human data
    human_data = np.load(human_dir)
    human_data_resh = human_data.reshape(28,8)

    for init in init_strings:
        
        # load the model data and reshape
        model_data = np.load(nn_dir + init + ".npy")
        model_data_resh = model_data.reshape(28,8)

        # calculate CC for each epoch and block and append to dict
        for epoch in range(human_data_resh.shape[1]):

            # normalise the data
            # norm_human = human_data_resh[:,epoch]/np.sum(human_data_resh[:,epoch])
            # norm_model = model_data_resh[:, epoch]/np.sum(model_data_resh[:, epoch])

            # this is the result
            res = cross_entropy(human_data_resh[:,epoch], model_data_resh[:, epoch])

            # append to dict
            if init not in ce_dict:
                ce_dict[init] = []
            ce_dict[init].append(res)
        
        # flatten the data
        human_data_flat = human_data_resh.reshape(224)
        model_data_flat = model_data_resh.reshape(224)

        # normalise
        human_data_flat_norm = human_data_flat/np.sum(human_data_flat)
        model_data_flat_norm = model_data_flat/np.sum(human_data_flat)

        # result for all
        res_all = cross_entropy(model_data_flat_norm, human_data_flat_norm)

        ce_dict["overall"].append(res_all)

    return ce_dict

ce_dict = calculate_CEs(init_strings, human_dir, nn_dir)
# cc_dict = calculate_CCs(init_strings, human_dir, nn_dir)


def plot_CEs(cc_dict, init_strings, title, save_path):
    # markers = ['o', 'h', 'x', 'd', 'p', '^']
    blocks = np.arange(len(cc_dict["1"]))
    colors = plt.cm.Reds_r(np.linspace(.25,1,len(cc_dict["1"])))

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.9), dpi=300, facecolor='w')

    for i, init in enumerate(init_strings):
        ax.plot(blocks, cc_dict[init], color = colors[i], label = init, marker = '.', linewidth=2.5)
    
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
    ax.set_ylabel("Cross Entropy $H(Human, Network)$", fontsize = 10)

    # make the legend
    ax.legend(title = 'Variance $\sigma^2$', bbox_to_anchor=(0.95,0.6))

    # make the title
    ax.set_title(title, fontweight='bold', y=1.05)
    # plt.subplots_adjust(bottom=0.15, left=.15, right=.15)
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_path)

plot_CEs(ce_dict, 
        init_strings,
        "Human-Linear(Sigmoid) Network CEs by Initialisation",
        os.getcwd() + "/figures_crossentropy/clip/Linear_sig_averaged.png"
        )

# "Human-Linear(Sigmoid) Network CCs by Initialisation"
# "/figures_correlations/Correlations_linear_sig_averaged.svg"

def plot_global_CEs(cc_list, init_strings, title, save_path):
    blocks = np.arange(len(cc_list))

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.9), dpi=300, facecolor='w')

    ax.plot(blocks, cc_list, marker = 'o', color='slateblue', linewidth=2.5)
    
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
    ax.set_ylabel("Cross Entropy $H(Human, Network)$", fontsize = 10)

    # make the title
    ax.set_title(title, fontweight='bold', y=1.05)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.15, left=.3, right=.9)

    plt.show()
    # plt.savefig(save_path)

plot_global_CEs(ce_dict["overall"], 
                init_strings, 
                "Overall Human-Linear(Sigmoid) Network CEs by Initialisation",
                os.getcwd() + "/figures_crossentropy/clip/Linear_sig_overall_averaged.png")

# "Overall Human-Linear(Sigmoid) Network CCs by Initialisation",
# os.getcwd() + "/figures_correlations/Correlations_linear_sig_overall_averaged.svg")