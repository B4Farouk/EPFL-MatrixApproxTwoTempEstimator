from numpy import extract, infty
import globals as g
import theory

def extract_param(line, param_key):
    if param_key >= 0:
        return float(line[1:].split(g.CFG_VAL_SEP)[param_key])
    
    splitted = line[1:].split(g.CFG_VAL_SEP)
    if(param_key == -1): #requesting sigma_u * sigma_v
        return float(splitted[g.key_u_i_std]) * float(splitted[g.key_v_i_std])
    elif(param_key == -2): #requesting beta_u / beta_v
        beta_v = float(splitted[g.key_inv_temp_v])
        if(beta_v == 0): 
            return -1 #invalid value
        
        beta_u = float(splitted[g.key_inv_temp_u])
        
        return beta_u / beta_v

    return -1 #invalid value

def parse_data(config_path, res_path, x_key, legend_params_keys):
    x_vals  = []
    
    outputs = [[] for i in range(0, g.OUTPUT_SIZE)]
    
    legend_params_vals = []
    
    alpha_u_vals = []
    alpha_v_vals = []
    alpha_vals = (alpha_u_vals, alpha_v_vals)
    
    beta_u_vals = []
    beta_v_vals = []
    beta_vals = (beta_u_vals, beta_v_vals)
    
    ## extracting parameter values
    with open(config_path, g.PARSED_OPEN_MODE) as config_file:
        for line in config_file:
            if(not line.startswith(g.CFG_LINE_IND)):
                continue
            
            x_vals.append(extract_param(line, x_key))
            
            ##read alpha values for rectification, if needed
            alpha_u_vals.append(extract_param(line, g.key_alpha_u))
            alpha_v_vals.append(extract_param(line, g.key_alpha_v))
            
            ##read beta values for 3D plot
            if x_key == -3:
                beta_u_vals.append(extract_param(line, g.key_inv_temp_u))
                beta_v_vals.append(extract_param(line, g.key_inv_temp_v))
            
            for param_key_str in legend_params_keys:
                legend_params_vals.append(round(extract_param(line, int(param_key_str)), 4))
                
        config_file.close()
    
    ## extracting MSEs
    with open(res_path, g.PARSED_OPEN_MODE) as res_file:
        for line in res_file:
            splitted = line.split(g.CFG_VAL_SEP)
            
            for index in range(0, g.OUTPUT_SIZE):
                outputs[index].append(float(splitted[index]))
        res_file.close()
    
    return x_vals, outputs, legend_params_vals, alpha_vals, beta_vals

def parse_MMSE_fcts(config_path):
    theo_MMSE_functions = []
    
    with open(config_path, g.PARSED_OPEN_MODE) as config_file:
        for line in config_file:
            if(not line.startswith(g.CFG_LINE_IND)):
                continue
            
            theo_MMSE_functions.append(
                theory.theoretical_MMSE_function(
                    extract_param(line, g.key_alpha_u), 
                    extract_param(line, g.key_alpha_v),
                    extract_param(line, g.key_u_i_std_gt)**2,
                    extract_param(line, g.key_u_i_std)**2,
                    extract_param(line, g.key_v_i_std_gt)**2,
                    extract_param(line, g.key_v_i_std)**2
                )
            )
            
        config_file.close()
            
    return theo_MMSE_functions

def parse_MMSE_fcts_sig_prod(config_path):
    theo_MMSE_functions = []
    
    with open(config_path, g.PARSED_OPEN_MODE) as config_file:
        for line in config_file:
            if(not line.startswith(g.CFG_LINE_IND)):
                continue
            
            theo_MMSE_functions.append(
                theory.theoretical_MMSE_fct_of_sigma_product(
                    extract_param(line, g.key_alpha_v),
                    extract_param(line, g.key_snr_gt),
                    extract_param(line, g.key_snr),
                    extract_param(line, g.key_u_i_std_gt),
                    extract_param(line, g.key_v_i_std_gt)
                )
            )
        config_file.close()
    
    return theo_MMSE_functions

def parse_plt_opts(options, option_name):
    for option in options:
        if option == option_name or option == g.PLOT_OPTIONS[4]:
            return True
        
    print("parsers.py:plt_opt_parser: option " + option_name + " not found.")
    return False

def parse_cfg_line(line):
    tokens = line[1:].split(g.CFG_VAL_SEP)
                
    ## create the configuration
    config = {}
    for key in g.param_keys:
        if key >= 0: 
            config.update({key : float(tokens[key])})
        
    return config

def parse_plt_desc(descriptor_path):  
    names = []
    x_key = None
    legend_param_keys = []
    
    with open(descriptor_path, g.PARSED_OPEN_MODE) as descriptor:
        for line in descriptor:
            if len(line) == 0 or line == "\n": continue
            if line[0] == "#": continue 
            
            splitted = line.split(g.DESCRIPTOR_SEP)
            
            if line.startswith(g.DESCRIPTOR_PARAMS_LINE_IND):
                x_key = int(splitted[0][1:])
                for i in range(1, len(splitted)):
                    legend_param_keys.append(int(splitted[i]))
            else:
                names.append((splitted[0].strip(), splitted[1].strip()))        
        descriptor.close()
             
    return names, x_key, legend_param_keys

def should_rectify_MSEs(options):
    for option in options: 
        if option == "--rectify":
            print("RECTIFIED")
            return True 
    
    print("NOT RECTIFIED")
    return False
