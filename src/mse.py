import sys

import multiprocessing as mp

import numpy as np
import numpy.random as rand
import math

import globals as g 
import parsers

import copy

#######################################################################
# AUXILARY FUNCTIONS
#######################################################################

norm = np.linalg.norm
rand_std_normal = rand.standard_normal

def normal_iid_vect(vect_dim, vect_shape, mean_i, var_i):
    assert var_i > 0
    return np.reshape(rand.multivariate_normal(mean_i * np.ones((vect_dim, ), dtype = float), var_i * np.eye(N = vect_dim, dtype = float)), vect_shape)

def dhamiltonian_du(u, v, Y_gt, snr_to_n, u_i_var):
    return - snr_to_n * (Y_gt - snr_to_n * (u @ v.T)) @ v + (u / u_i_var)

def dhamiltonian_dv(u, v, Y_gt, snr_to_n, v_i_var):
    return - snr_to_n * ((Y_gt - snr_to_n * (u @ v.T))).T @ u + (v / v_i_var)

#######################################################################
# MSE COMPUTATION
#######################################################################

def mse(config):    
    ## Model simple parameters
    gt_dataset_size = int(config.get(g.key_gt_dataset_size))
    
    n               = int(config.get(g.key_n))
    alpha_u         = config.get(g.key_alpha_u)
    alpha_v         = config.get(g.key_alpha_v)
    
    snr_gt          = config.get(g.key_snr_gt)
    snr             = config.get(g.key_snr)
    
    lr_u            = config.get(g.key_lr_u) 
    lr_v            = config.get(g.key_lr_v)
    
    inv_temp_u      = config.get(g.key_inv_temp_u)
    inv_temp_v      = config.get(g.key_inv_temp_v)
    
    u_i_mean_gt     = config.get(g.key_u_i_mean_gt)
    u_i_mean        = config.get(g.key_u_i_mean)
    v_i_mean_gt     = config.get(g.key_v_i_mean_gt)
    v_i_mean        = config.get(g.key_v_i_mean)
    
    u_i_std_gt      = config.get(g.key_u_i_std_gt)
    u_i_std         = config.get(g.key_u_i_std)
    v_i_std_gt      = config.get(g.key_v_i_std_gt)
    v_i_std         = config.get(g.key_v_i_std)
    
    sampling_threshhold = int(config.get(g.key_sampling_threshhold))
    nb_samples          = int(config.get(g.key_nb_samples))
    delta_t             = config.get(g.key_delta_t)
    
    ## Checks
    assert n > 0.0 and alpha_u > 0.0 and alpha_v > 0.0
    assert u_i_std_gt > 0.0 and v_i_std_gt > 0.0 and u_i_std > 0.0 and v_i_std > 0.0 
    assert snr_gt > 0.0 and snr > 0.0
    assert lr_u > 0.0 and lr_v > 0.0
    assert inv_temp_u > 0.0 and inv_temp_v > 0.0
    assert sampling_threshhold >= 0
    assert nb_samples > 0
    assert delta_t > 0.0
        
    ## Model: Composite Parameters 
    u_dim = int(n * alpha_u)
    v_dim = int(n * alpha_v)

    u_i_var = u_i_std**2
    v_i_var = v_i_std**2
    
    u_i_var_gt = u_i_std_gt**2
    v_i_var_gt = v_i_std_gt**2

    snr_gt_to_n = math.sqrt(snr_gt / n)
    snr_to_n = math.sqrt(snr / n)

    inv_lr_u = 1.0 / lr_u 
    inv_lr_v = 1.0 / lr_v

    tlrr_u = math.sqrt(2.0 / (inv_temp_u * inv_lr_u)) 
    tlrr_v = math.sqrt(2.0 / (inv_temp_v * inv_lr_v))

    assert u_dim > 0 and v_dim > 0 
    assert u_i_var > 0.0 and v_i_var > 0.0
    assert snr_gt_to_n > 0.0 and snr_to_n > 0.0 
    assert tlrr_u > 0.0 and tlrr_v > 0.0

    ## Sampling parameters 
    nb_iter = sampling_threshhold + nb_samples
    
    ## Shapes
    u_shape = (u_dim, 1) 
    v_shape = (v_dim, 1)

    gt_data_shape = (u_dim, v_dim)

    u_samples_array_shape = (nb_samples, u_dim, 1)
    v_samples_array_shape = (nb_samples, v_dim, 1)
    uvT_samples_array_shape = (nb_samples, u_dim, v_dim)
    
    ## Constants
    matrix_mse_normalizing_cst = u_dim * v_dim * gt_dataset_size 
    vector_u_mse_normalizing_cst = u_dim * gt_dataset_size 
    vector_v_mse_normalizing_cst = v_dim * gt_dataset_size
    vector_norm_normalizing_cst = math.sqrt(n)
    
    ####################
    ## MSE COMPUTATION
    ####################
    
    ## Init MSEs
    type0_MSE_u = 0.0
    type0_MSE_v = 0.0
    type1_MSE_u = 0.0
    type1_MSE_v = 0.0
    MSE_uvT = 0.0
    
    ## Array that will hold the variances of uvT_samples, one at each experiment
    var_uvT_samples_array = np.array([0.0]*gt_dataset_size, dtype=float)
    
    ## Arrays that will hold the norms of u(t) and v(t) for every dynamics' discrete step (every iteration)
    u_t_norms_array = np.array([0.0]*nb_iter, dtype=float)
    v_t_norms_array = np.array([0.0]*nb_iter, dtype=float)
    
    ## Arrays that will hold the average norm of u(t) and average norm of v(t), one value per experiment
    avg_norm_u_t_array = np.array([0.0]*gt_dataset_size, dtype=float)
    avg_norm_v_t_array = np.array([0.0]*gt_dataset_size, dtype=float)

    for i in range(0, gt_dataset_size):
        ## Generating ground truth the data
        u_gt = normal_iid_vect(u_dim, u_shape, u_i_mean_gt, u_i_var_gt)
        v_gt = normal_iid_vect(v_dim, v_shape, v_i_mean_gt, v_i_var_gt)
        
        Z = rand_std_normal(gt_data_shape)
        uvT_gt = u_gt @ v_gt.T
        Y_gt = snr_gt_to_n * uvT_gt + Z

        assert u_gt.shape == u_shape
        assert v_gt.shape == v_shape
        assert uvT_gt.shape == gt_data_shape
        assert Y_gt.shape == gt_data_shape
        assert Z.shape == gt_data_shape
    
        ## Sampling: Initialization
        u = normal_iid_vect(u_dim, u_shape, u_i_mean, u_i_var)
        v = normal_iid_vect(v_dim, v_shape, v_i_mean, v_i_var)

        assert u.shape == u_shape
        assert v.shape == v_shape
    
        u_samples = np.array(np.zeros(u_samples_array_shape))
        v_samples = np.array(np.zeros(v_samples_array_shape))
        uvT_samples = np.array(np.zeros(uvT_samples_array_shape))
    
        assert u_samples.size == nb_samples * u_dim 
        assert v_samples.size == nb_samples * v_dim

        ## Sampling: Descritized Langevin Dynamics
        for j in range(0, nb_iter):        
            w_u = rand_std_normal(u_shape)
            w_v = rand_std_normal(v_shape)
        
            assert w_u.shape == u_shape
            assert w_v.shape == v_shape 
    
            u = u + (-lr_u * dhamiltonian_du(u, v, Y_gt, snr_to_n, u_i_var) + tlrr_u * w_u) * delta_t 
            v = v + (-lr_v * dhamiltonian_dv(u, v, Y_gt, snr_to_n, v_i_var) + tlrr_v * w_v) * delta_t

            u_t_norms_array[j] = norm(u)
            v_t_norms_array[j] = norm(v)
            
            if(j >= sampling_threshhold):
                index = j - sampling_threshhold 
                u_samples[index] = u
                v_samples[index] = v
    
        for k in range(0, nb_samples):
            uvT_sample = u_samples[k] @ v_samples[k].T 
            uvT_samples[k] = uvT_sample
    
        ## Monte Carlo estimation 
        u_avg = np.mean(u_samples, axis = 0)
        v_avg = np.mean(v_samples, axis = 0)
        uvT_avg = np.mean(uvT_samples, axis = 0)
        
        assert u_avg.shape   == u_gt.shape
        assert v_avg.shape   == v_gt.shape
        assert uvT_avg.shape == uvT_gt.shape
        
        ## Mean Variance uvT samples at the current experiment
        matrix_normes_array = norm(uvT_samples - uvT_avg, axis = (1, 2))
        assert matrix_normes_array.shape == (nb_samples, )
        var_uvT_samples_array[i] = np.mean(matrix_normes_array**2)
    
        ## Unnormalized MSEs
        type0_MSE_u += norm(u_gt - u_avg)**2
        type0_MSE_v += norm(v_gt - v_avg)**2
        
        type1_MSE_u += min(norm(u_gt - u_avg)**2, norm(u_gt + u_avg)**2)
        type1_MSE_v += min(norm(v_gt - v_avg)**2, norm(v_gt + v_avg)**2)
        
        MSE_uvT += norm(uvT_gt - uvT_avg)**2
        
        ## Average of all u(t) and v(t) norms tracked in the dynamics
        avg_norm_u_t_array[i] = np.mean(u_t_norms_array)
        avg_norm_v_t_array[i] = np.mean(v_t_norms_array)
        
    ## Normalization of MSEs
    type0_MSE_u /= vector_u_mse_normalizing_cst
    type0_MSE_v /= vector_v_mse_normalizing_cst
    type1_MSE_u /= vector_u_mse_normalizing_cst
    type1_MSE_v /= vector_v_mse_normalizing_cst
    MSE_uvT /= matrix_mse_normalizing_cst
    
    ## Mean of ( Mean Variance of uvT samples per experiment )
    mean_var_uvT_samples = np.mean(var_uvT_samples_array)
    
    ## Mean over all experiments of the average norm of u(t) and v(t) tracked in the dynamics
    mean_norm_u_t = np.mean(avg_norm_u_t_array) / vector_norm_normalizing_cst
    mean_norm_v_t = np.mean(avg_norm_v_t_array) / vector_norm_normalizing_cst
    
    ## NEVER CHANGE THE ORDER OF THESE ELEMENTS IN THIS LIST OR ELSE THIS BREAKS BACKWARD COMPATIBILITY:
    ## IF A NEW ELEMENT IS ADDED, IT SHOULD BE AT THE END OF THE LIST
    ## PARSER EXPECTS THE RESUTLS TO BE WRITTEN IN THIS ORDER, IF THE ORDER CHANGES PARSING IS INCORRECT
    ## PLOTTER EXPECTS THE RESULTS TO BE PARSED AND GROUPED BY CATHEGORY IN THIS ORDER, IF ORDER CHANGES PLOTTING IS INCORRECT
    out = list([type0_MSE_u, type0_MSE_v, MSE_uvT, mean_var_uvT_samples, mean_norm_u_t, mean_norm_v_t, type1_MSE_u, type1_MSE_v])
    return out

def parallelized_mse(config, proc_pool_size):
    
    # Divide
    gt_dataset_size = int(config.get(g.key_gt_dataset_size))
    
    chunk_size = gt_dataset_size // proc_pool_size              # euclidean division of gt_dataset_size by proc_pool_size
    remainder = gt_dataset_size - chunk_size * proc_pool_size   # euclidean division: 0 <= remainder < proc_pool_size
    
    assert 0 <= remainder and remainder < proc_pool_size
    
    configs_list_length = 0
    configs_list = None
    partitions = None
    
    if chunk_size == 0: # case there are less experiments than the initial size of the pool of processes
        configs_list = [copy.deepcopy(config) for i in range(0, gt_dataset_size)]
        configs_list_length = len(configs_list)
        
        for cfg in configs_list: 
            cfg.update({g.key_gt_dataset_size : 1})
    
        partitions = [1]*gt_dataset_size
        proc_pool_size = gt_dataset_size ## no need for more processes
    
    else:
        configs_list = [copy.deepcopy(config) for i in range(0, proc_pool_size)]
        configs_list_length = len(configs_list)
        
        partitions = [0]*configs_list_length
        
        partition_size = 0
        for cfg_idx, cfg in enumerate(configs_list):
            if remainder > 0:
                partition_size = chunk_size + 1
                remainder -= 1
            else:
                partition_size = chunk_size
            
            cfg.update({g.key_gt_dataset_size : partition_size})
            partitions[cfg_idx] = partition_size
        
    # Conquer in parallel
    proc_pool = mp.Pool(proc_pool_size)
    outputs = None
    
    try:
        outputs = proc_pool.map(mse, configs_list, chunksize=1)
    except Exception as error:
        proc_pool.close()
        print(error)
        sys.exit(-1)
    
    proc_pool.close()
    
    # Combine
   
    output = [0.0] * g.OUTPUT_SIZE
    
    ## output length is the same for all outputs since they result from the call to the same function
    for cfg_idx, an_output in enumerate(outputs):
        for i in range(0, g.OUTPUT_SIZE):
            output[i] += an_output[i] * (partitions[cfg_idx] / gt_dataset_size)

    return output

#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":
    proc_pool_size = int(sys.argv[1])
    config_path = g.CONFIG_PATH + sys.argv[2]
    res_path = g.RES_PATH + sys.argv[3]
    
    with open(res_path, g.OUT_FILE_OPEN_MODE) as res_file:
        with open(config_path, g.CONFIG_FILE_OPEN_MODE) as config_file:
            
            for line in config_file:
                
                ## if the line is not a configuration skip it
                if(not line.startswith(g.CFG_LINE_IND)): 
                    continue
             
                config = parsers.parse_cfg_line(line)
            
                ## compute MSE for the given configuration
                output = parallelized_mse(config, proc_pool_size)
                
                ## write the results to the output file
                res_file.write(g.CFG_VAL_SEP.join([str(out) for out in output]) + '\n')
                
        config_file.close()
    res_file.close()
