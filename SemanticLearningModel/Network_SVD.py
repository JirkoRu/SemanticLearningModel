import numpy as np
import pandas as pd
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from matplotlib.pyplot import ylabel
from matplotlib.font_manager import FontProperties
from preprocess_in_out import sort_matrices

nn_dir = os.getcwd() + "/linear_init_exp_48rep/saved_inputs-outputs/lr_"

# substrings to load each initialisation
init_strings = ["1", "0.1", "0.01", "0.001", "0.0001", "1e-05"]

inputs = np.load(nn_dir + init_strings[4] + "/" + "inputs_run_0.npy")
outputs = np.load(nn_dir + init_strings[4] + "/" + "outputs_run_0.npy")

sorted_inputs, sorted_outputs = sort_matrices(inputs, outputs)
sorted_outputs = np.swapaxes(sorted_outputs,0,1)
sorted_outputs = np.swapaxes(sorted_outputs,0,2)
print(sorted_outputs.shape)
u, s, vt = np.linalg.svd(sorted_outputs, full_matrices=False)
print(u.shape, s.shape, vt.shape)

# lets create a function for plotting singular values

from turtle import color


def plot_singular_values(s, type, fig_name, title, locs=False):
    """a function to plot my singular values"""
    x_vals = np.arange(s.shape[0]) # lets get the x vals

    # make the annotation labels
    annos = ['$a_1$', '$a_2$', '$a_{3,4}$' ]

    # get the singular values in proper form
    s[:,2] = (s[:,2] + s[:,3])/2
    s_combined = np.delete(s,3,1)
    colors = plt.cm.Reds_r(np.linspace(0, .7, 3))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300, facecolor='w')
    for i in range(s_combined.shape[1]):

        ax.plot(x_vals, s_combined[:,i], color=colors[i], linewidth=2, label=annos[i])

        if locs: 
            ax.text(locs[i][0],locs[i][1], annos[i], fontsize= 19)

    ax.set_xlabel(' t(%s)' %(type), fontsize= 20)
    ax.set_ylabel('$a_i(t)$', fontsize= 18)

    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    # title
    ax.set_title(title, y=1.05)
    
    # legend
    ax.legend(title = 'Singular Value')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(fig_name)
    
plot_singular_values(s, 'Epochs', 
                    'SVD_plots/singular_values_linear_run_0_0.0001.png',
                    "SVD Linear Network $\sigma^2$ = .0001")