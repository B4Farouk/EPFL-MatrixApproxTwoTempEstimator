import sys

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

import globals as g
import parsers

############################################################
# AUXILARY FUNCTIONS
############################################################

def initPlot(title, x_label, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

def showPlot():
    plt.show()

def plotFor(x_vals, y_vals, z_vals):
    z_vals = np.array(z_vals)
    
    colmap = cm.ScalarMappable(cmap=cm.viridis)
    colmap.set_array(z_vals)
    
    plt.colorbar(colmap)
    plt.scatter(x_vals, y_vals, c=z_vals, marker='o')

############################################################
# MAIN
############################################################

if __name__ == "__main__":
    
    ## read program arguments
    plt_descriptor_path = g.DESCRIPTOR_PATH + sys.argv[1]
    filenames, x_key, legend_params_keys = parsers.parse_plt_desc(plt_descriptor_path)
        
    ## get options
    options = None 
    try: 
        options = sys.argv[2:]
    except: 
        pass
    
    rectify_MSEs = parsers.should_rectify_MSEs(options)
    
    ## parse files
    ### data is a list of tuples (x_vals, outputs, legend_param_vals, alpha_vals, beta_vals)
    ### outputs order: type0_MSE_u, type0_MSE_v, MSE_uvT, mean_var_uvT_samples, mean_norm_u_t, mean_norm_v_t, type1_MSE_u, type1_MSE_v
    ### alpha_vals is a tuple of alpha_u list and alpha_v list, read here, to normalize properly and recover from Farzad's mistake
    ### beta_vals is a tuple of beta_u list and beta_v list to perform the 3D MSE plot
    data = [parsers.parse_data(g.CONFIG_PATH + config_name, g.RES_PATH + res_name, x_key, legend_params_keys) for config_name, res_name in filenames]
    
    initPlot("MSE scatter heatmap", g.param_name_map.get(g.key_inv_temp_u), g.param_name_map.get(g.key_inv_temp_v))
    
    for _, outputs, _, alpha_vals, beta_vals in data:
        if x_key == -3:
            beta_u_vals, beta_v_vals = beta_vals
            if rectify_MSEs:
                alpha_u_vals, alpha_v_vals = alpha_vals
                plotFor(beta_u_vals, beta_v_vals, np.array(outputs[2])/(np.array(alpha_u_vals) * np.array(alpha_v_vals)))
            else:
                plotFor(beta_u_vals, beta_v_vals, outputs[2])
    
    showPlot()