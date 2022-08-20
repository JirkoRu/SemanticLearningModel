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

nn_dir = os.getcwd() + "/linear_init_exp/reduced_inputs-outputs_100/linear_small_weights_16hidden_outputs_lr_"
human_dir = "C:/Users/Jirko/Desktop/Hip_Lab/analysis_scripts/data_loading/data_v2_quotes/in_out_matrix.npy"
init_strings = ["1", "0.1", "0.01", "0.001", "0.0001", "1e-05"]

human_data = np.load(human_dir)
model_data = np.load(nn_dir + init_strings[0] + ".npy")

human_data_flat = human_data.reshape(28,8)
model_data_flat = model_data.reshape(28,8)

print(human_data_flat.shape)
sum_cc = 0
for epoch in range(human_data_flat.shape[1]):
    res = stats.pearsonr(human_data_flat[:,epoch], model_data_flat[:, epoch])
    sum_cc+= res[0]
    print(epoch)
    print(res)

print(sum_cc/human_data_flat.shape[1])

human_data_flat = human_data_flat.reshape(224)
model_data_flat = model_data_flat.reshape(224)

res = stats.pearsonr(human_data_flat, model_data_flat)
print(res)



