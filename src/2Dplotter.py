import sys

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

import globals as g
import parsers

############################################################
# GLOBALS
############################################################

#if beta experiment, this needs to be adapted
SIGMA_U = 0.75
SIGMA_V = 0.75

NB_POINTS = 256
START_OF_SPACE = 0
END_OF_SPACE = 2

############################################################
# AUXILARY FUNCTIONS
############################################################

def generateParamsLegendStr(legend_params_keys, legend_params_vals):
    param_str = ""
    for legend_param_key, legend_param_val in zip(legend_params_keys, legend_params_vals):
            param_str = param_str + g.param_name_map[legend_param_key] + "=" + str(legend_param_val) + ' '
    
    return param_str

def initPlot(plot_title, y_axis_title, x_key):
    plt.title(plot_title + " as function of " + g.param_name_map.get(x_key)) 
    plt.ylabel(y_axis_title)
    plt.xlabel(g.param_name_map.get(x_key))

def plotFor(x_vals, values, label, ls='-'):
    plt.plot(x_vals, values, label = label, ls=ls, marker = 'o', markersize=0)

def showPlot():
    plt.legend()
    plt.show()

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
    data = [parsers.parse_data(g.CONFIG_PATH + config_name, g.RES_PATH + res_name, x_key, legend_params_keys) for config_name, res_name in filenames]
    
    ## plotting vector MSEs: type 0
    initPlot("MSE(type 0) of u and v", "MSE(type 0)", x_key)
    
    for x_vals, outputs, legend_params_vals, alpha_vals, _ in data:
        param_str = generateParamsLegendStr(legend_params_keys, legend_params_vals)
        if rectify_MSEs:
            alpha_u_vals, alpha_v_vals = alpha_vals
            plotFor(x_vals, np.array(outputs[0]) / np.array(alpha_u_vals), "u MSE(type 0): " + param_str)
            plotFor(x_vals, np.array(outputs[1]) / np.array(alpha_v_vals), "v MSE(type 0): " + param_str)
        else: 
            plotFor(x_vals, outputs[0], "u MSE(type 0): " + param_str)
            plotFor(x_vals, outputs[1], "v MSE(type 0): " + param_str)
    
    showPlot()
    
    ## plotting vector MSEs: type 1
    try:
        if parsers.parse_plt_opts(options, g.PLOT_OPTIONS[3]):
            initPlot("MSE(type 1) of u and v", "MSE(type 1)", x_key)
            
            for x_vals, outputs, legend_params_vals, alpha_vals, _ in data:
                param_str = generateParamsLegendStr(legend_params_keys, legend_params_vals)
                if rectify_MSEs:
                    alpha_u_vals, alpha_v_vals = alpha_vals
                    plotFor(x_vals, np.array(outputs[6]) / np.array(alpha_u_vals), "u MSE(type 1): " + param_str)
                    plotFor(x_vals, np.array(outputs[7]) / np.array(alpha_v_vals), "v MSE(type 1): " + param_str)
                else: 
                    plotFor(x_vals, outputs[6], "u MSE(type 1): " + param_str)
                    plotFor(x_vals, outputs[7], "v MSE(type 1): " + param_str)                    
            
            showPlot()
    except Exception as e:
        print(e)
    
    ## plotting matrix MSE
    initPlot("Mean Squared Errors", "MSE", x_key)
    
    for x_vals, outputs, legend_params_vals, alpha_vals, _ in data:        
        param_str = generateParamsLegendStr(legend_params_keys, legend_params_vals)
        if rectify_MSEs: 
            alpha_u_vals, alpha_v_vals = alpha_vals
            plotFor(x_vals, np.array(outputs[2]) / (np.array(alpha_u_vals) * np.array(alpha_v_vals)), "X MSE: " + param_str)
        else: 
            alpha_u_vals, alpha_v_vals = alpha_vals
            plotFor(x_vals, outputs[2], "X MSE: " + param_str)
            
    ## MMSE option
    try:
        if parsers.parse_plt_opts(options, g.PLOT_OPTIONS[1]):
            space = np.linspace(START_OF_SPACE, END_OF_SPACE, NB_POINTS)
            MMSE_fcts_parser = None
            cst_space = None #TD
            
            if x_key >= 0:
                MMSE_fcts_parser = parsers.parse_MMSE_fcts   
            elif x_key == -1:
                MMSE_fcts_parser = parsers.parse_MMSE_fcts_sig_prod
            elif x_key == -2: #TD
                prod_sigs = SIGMA_U * SIGMA_V
                cst_space = np.array([prod_sigs]*NB_POINTS)
                MMSE_fcts_parser = parsers.parse_MMSE_fcts_sig_prod

            legend_params_vals_list = [legend_params_vals for _, _, legend_params_vals, _, _ in data]
            MMSE_fcts_lists = [MMSE_fcts_parser(g.CONFIG_PATH + config_name) for config_name ,_ in filenames]
            
            zipped = zip(legend_params_vals_list, MMSE_fcts_lists)
            
            for legend_params_vals, MMSE_fcts_list in zipped:
                param_str = generateParamsLegendStr(legend_params_keys, legend_params_vals)
                legend_str = None 
                if x_key == -1: 
                    legend_str = "X theoretical mismatched MSE: " + param_str
                elif x_key >= 0: 
                    legend_str = "X theoretical MMSE: " + param_str
                #for now we plot the first one for every config file
                #for f in MMSE_fcts_list: 
                if x_key == -2: #TD
                    plotFor(space, MMSE_fcts_list[0](cst_space), legend_str, ls="--")
                else:
                    plotFor(space, MMSE_fcts_list[0](space), legend_str, ls="--")                  
    except Exception as e:
        print(e)
    
    showPlot()
    
    ## plotting other data
    try:
        if parsers.parse_plt_opts(options, g.PLOT_OPTIONS[0]):
            initPlot("Average Sample Variance of X", "Average Sample Variance", x_key)
            for x_vals, outputs, legend_params_vals, _, _ in data:
                param_str = generateParamsLegendStr(legend_params_keys, legend_params_vals)
                plotFor(x_vals, outputs[3], "X samples average mean variance: " + param_str)
            showPlot()
            
        if parsers.parse_plt_opts(options, g.PLOT_OPTIONS[2]):
            initPlot("Mean Norm of u(t) and v(t)", "Mean Norm", x_key)
            for x_vals, outputs, legend_params_vals, _, _ in data:
                param_str = generateParamsLegendStr(legend_params_keys, legend_params_vals)
                plotFor(x_vals, outputs[4], "||u(t)|| mean: " + param_str)
                plotFor(x_vals, outputs[5], "||v(t)|| mean: " + param_str)
            showPlot()
    except Exception as e:
        print(e)
    
