####################################################################
# I/O & PARSING
####################################################################

CONFIG_PATH = "./configs/"
RES_PATH = "./res/"
DESCRIPTOR_PATH = "./plt_descriptors/"

CONFIG_FILE_OPEN_MODE = 'r'
OUT_FILE_OPEN_MODE = 'w'

PARSED_OPEN_MODE = 'r'

####################################################################
# CONFIGURATION FORMAT
####################################################################
'''
Configurations File Format: 
    > Each line corresponds to one configuration for which to compute the MSEs of u and v and X
    > A configuration has the following format (all parameters must be specified otherwise the process exits)
        :gt_dataset_size|n|alpha_u|alpha_v|snr_gt|snr|lr_u|lr_v|inv_temp_u|inv_temp_v|u_i_mean_gt|u_i_mean|v_i_mean_gt|v_i_mean|u_i_std_gt|u_i_std|v_i_std_gt|v_i_std|sampling_threshhold|nb_samples|delta_t
    > Each line starting with the character ':' is a configuration
    > Each line starting with $ specifies a runnable section of a configuration file
    
    * Model Parameters
    gt_dataset_size: the number of experiments
    n              : parameter n
    alpha_u        : parameter alpha_uu_i_mean
    alpha_v        : parameter alpha_v
    snr_gt         : ground truth of signal to noise ratio
    snr            : assumed signal to noise ratio
    lr_u           : learning rate of u
    lr_v           : learning rate of v 
    inv_temp_u     : inverse temperature beta_1
    inv_temp_v     : inverse temperature beta_2
    u_i_mean_gt    : ground truth mean of each component of u
    u_i_mean       : mean of each component of u
    v_i_mean_gt    : ground truth mean of each component of v
    v_i_mean       : mean of each component of v
    u_i_std_gt     : ground truth standard deviation of each component of u
    u_i_std        : assumed standard deviation of each component of u
    v_i_std_gt     : ground truth standard deviation of each component of v
    v_i_std        : assumed standard deviation of each component of v
    
    * Sampling Parameters
    sampling_threshhold: the threshhold after which sampling starts
    nb_samples         : number of samples
    delta_t            : the hamiltonian dynamics' step
'''

CFG_LINE_IND = ":"

CFG_VAL_SEP = "|"

DESCRIPTOR_PARAMS_LINE_IND = "$"
DESCRIPTOR_SEP = ":"

####################################################################
# PARAMETER UTILITY
####################################################################

key_gt_dataset_size = 0
key_n               = 1

key_alpha_u         = 2
key_alpha_v         = 3

key_snr_gt          = 4
key_snr             = 5

key_lr_u            = 6
key_lr_v            = 7

key_inv_temp_u      = 8
key_inv_temp_v      = 9

key_u_i_mean_gt     = 10
key_u_i_mean        = 11
key_v_i_mean_gt     = 12
key_v_i_mean        = 13

key_u_i_std_gt      = 14
key_u_i_std         = 15
key_v_i_std_gt      = 16
key_v_i_std         = 17

key_sampling_threshhold = 18
key_nb_samples          = 19
key_delta_t             = 20

sigma_u_sigma_v_prod = -1
beta_u_beta_v_ratio = -2

param_keys = [    
    key_gt_dataset_size, 
    key_n,
    
    key_alpha_u,
    key_alpha_v,
    
    key_snr_gt,
    key_snr,
    
    key_lr_u,
    key_lr_v,
    
    key_inv_temp_u,
    key_inv_temp_v,
    
    key_u_i_mean_gt,
    key_u_i_mean,
    key_v_i_mean_gt,
    key_v_i_mean,
    
    key_u_i_std_gt,
    key_u_i_std,
    key_v_i_std_gt,
    key_v_i_std,
    
    key_sampling_threshhold,
    key_nb_samples,
    key_delta_t,
    
    sigma_u_sigma_v_prod, 
    beta_u_beta_v_ratio
]

param_name_map = {
    key_gt_dataset_size : "ground truth dataset size",
    key_n : "n",
    
    key_alpha_u : "⍺ᵤ",
    key_alpha_v : "⍺ᵥ",

    key_snr_gt : "λ*",
    key_snr : "λ",
    
    key_lr_u : "1/ηᵤ",
    key_lr_v : "1/ηᵥ",
    
    key_inv_temp_u : "βᵤ",
    key_inv_temp_v : "βᵥ",
    
    key_u_i_mean_gt : "𝜇*ᵤ",
    key_u_i_mean : "𝜇ᵤ",
    key_v_i_mean_gt : "𝜇*ᵥ",
    key_v_i_mean : "𝜇ᵥ",

    key_u_i_std_gt : "σ*ᵤ",
    key_u_i_std : "σᵤ",
    key_v_i_std_gt : "σ*ᵥ",
    key_v_i_std : "σᵥ",
    
    key_sampling_threshhold : "𝑇",
    key_nb_samples : "𝑆",
    key_delta_t : "Δt",
    
    sigma_u_sigma_v_prod : "σᵤσᵥ", 
    beta_u_beta_v_ratio : "βᵤ/βᵥ"
}

####################################################################
# MSE OUTPUT LENGTH
####################################################################

OUTPUT_SIZE = 8

####################################################################
# PLOT OPTIONS
####################################################################

## DO NOT CHANGE THE ORDER OF THESE ELEMENTS OTHERWISE THIS BREAKS BACKWARD COMPATIBILITY:
## IF YOU WANT TO ADD AN ELEMENT, ADD IT AT THE END OF THE ARRAY
PLOT_OPTIONS = [
    "--avg-var-samp", 
    "--theo-mmse",
    "--avg-vect-norms",
    "--mse-type1", 
    "--all"
]