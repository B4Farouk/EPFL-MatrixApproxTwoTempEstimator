import numpy as np
import numpy.random as rand
import torch 
import math

#######################################################################
# GLOBALS
#######################################################################

## Model: Simple Parameters 
n = 10.0
alpha_u = 1.0
alpha_v = 1.0

u_std_gt = 1.0
v_std_gt = 1.0
u_std = 1.0
v_std = 1.0

snr_gt = 1.0
snr = 1.0

lr_u = 1.0
lr_v = 1.0

inv_temp_u = 1.0
inv_temp_v = 1.0

assert n > 0.0 and alpha_u > 0.0 and alpha_v > 0.0
assert u_std_gt > 0.0 and v_std_gt > 0.0 and u_std > 0.0 and v_std > 0.0 
assert snr_gt > 0.0 and snr > 0.0
assert lr_u > 0.0 and lr_v > 0.0
assert inv_temp_u > 0.0 and inv_temp_v > 0.0 

## Model: Composite Parameters 
u_dim = int(n * alpha_u)
v_dim = int(n * alpha_v)

u_var = u_std**2
v_var = v_std**2

snr_gt_to_n = math.sqrt(snr_gt / n)
snr_to_n = math.sqrt(snr / n)

inv_lr_u = 1.0 / lr_u
inv_lr_v = 1.0 / lr_v 

tlrr_u = math.sqrt(2.0 / (inv_temp_u * lr_u)) 
tlrr_v = math.sqrt(2.0 / (inv_temp_v * lr_v))

assert u_dim > 0 and v_dim > 0 
assert u_var > 0.0 and v_var > 0.0
assert snr_gt_to_n > 0.0 and snr_to_n > 0.0 
assert tlrr_u > 0.0 and tlrr_v > 0.0

## Performance Evaluation Parameters
gt_dataset_size = 10

## Shapes
u_shape = (u_dim, 1) 
v_shape = (v_dim, 1)
data_shape = (u_dim, v_dim)

#######################################################################
# HELPERS
#######################################################################

def t_normal_iid_zero_mean_vect(vect_dim, vect_shape, std):
    assert std > 0
    return np.reshape(rand.multivariate_normal(np.zeros((vect_dim, ), dtype = float), (std**2) * np.eye(vect_dim, dtype = float)), vect_shape)

def multivariate_gaussian(vect_shape):
    return rand.standard_normal(vect_shape)

t_norm = torch.linalg.norm
t_mprod = torch.mm 
t_T = lambda vect: torch.transpose(vect, 0, 1)
np_norm = np.linalg.norm

def hamiltonian(u_t, v_t, Y_gt_t):
    uvT_t = t_mprod(u_t, t_T(v_t))
    return -0.5 * ( t_norm(Y_gt_t - snr_to_n * uvT_t)**2 + (t_norm(u_t)**2)/u_var + (t_norm(v_t)**2)/v_var )

#######################################################################
# GENERATING DATA
####################################################################### 

Y_gts = np.array([])
uvT_gts = np.array([])  
Z_s = np.array([])

## Generating the ground truth data 
for i in range(0, gt_dataset_size):
    u_gt = t_normal_iid_zero_mean_vect(u_dim, u_shape, u_std_gt)
    v_gt = t_normal_iid_zero_mean_vect(v_dim, v_shape, v_std_gt)
    Z = multivariate_gaussian(data_shape)
    uvT_gt = u_gt @ v_gt.T
    Y_gt = snr_gt_to_n * uvT_gt + Z

    uvT_gts = np.append(uvT_gts, uvT_gt)
    Z_s = np.append(Z_s, Z)
    Y_gts = np.append(Y_gts, Y_gt)

    assert u_gt.shape == u_shape
    assert v_gt.shape == v_shape
    assert Z.shape == data_shape
    assert Y_gt.shape == data_shape

Y_gts = np.reshape(Y_gts, (gt_dataset_size, u_dim, v_dim))
Z_s = np.reshape(Z_s, (gt_dataset_size, u_dim, v_dim))
uvT_gts = np.reshape(uvT_gts, (gt_dataset_size, u_dim, v_dim))

#######################################################################
# SAMPLING (using Discretized Langevin Dynamics Equations)
#######################################################################

u_s = np.array([]) 
v_s = np.array([])

## Sampling parameters 
nb_samples = 20
sampling_threshhold = 100
nb_iter = sampling_threshhold + nb_samples
delta_t = 1.0

# Sampling
for j in range(0, gt_dataset_size): 
    u_init = t_normal_iid_zero_mean_vect(u_dim, u_shape, u_std)
    v_init = t_normal_iid_zero_mean_vect(v_dim, v_shape, v_std)

    u = u_init; v = v_init 

    assert u_init.shape == u_shape 
    assert v_init.shape == v_shape 

    Y_gt = Y_gts[j]
    Y_gt_t = torch.tensor(Y_gt, dtype = float)

    assert Y_gt.shape == data_shape

    for i in range(0, nb_iter):
        w_u = multivariate_gaussian(u_shape)
        w_v = multivariate_gaussian(v_shape)

        u_t = torch.tensor(u, dtype = float, requires_grad = True)  
        v_t = torch.tensor(v, dtype = float, requires_grad = True)
        u_t.grad = None; v_t.grad = None

        hamiltonian(u_t, v_t, Y_gt_t).backward() # backward computation of gradient evaluated at u_t and v_t of the Hamiltonian function 

        grad_H_u = u_t.grad.numpy()
        grad_H_v = v_t.grad.numpy()
    
        u = u + (-inv_lr_u * grad_H_u + tlrr_u * w_u) * delta_t 
        v = v + (-inv_lr_v * grad_H_v + tlrr_v * w_v) * delta_t 

        assert w_u.shape == u_shape 
        assert w_v.shape == v_shape 

        if(i >= sampling_threshhold): 
            u_s = np.append(u_s, u)
            v_s = np.append(v_s, v)

u_s = np.reshape(u_s, (gt_dataset_size, nb_samples, u_dim))
v_s = np.reshape(v_s, (gt_dataset_size, nb_samples, v_dim))

assert u_s.size == gt_dataset_size * nb_samples * u_dim 
assert v_s.size == gt_dataset_size * nb_samples * v_dim

#######################################################################
# MATRIX RECONSTRUCTION
#######################################################################

u_avgs = np.mean(u_s, axis = 1)
v_avgs = np.mean(v_s, axis = 1)

assert u_avgs.shape == (gt_dataset_size, u_dim)
assert v_avgs.shape == (gt_dataset_size, v_dim) 

uvT_avgs = np.array([])

for i in range(0, gt_dataset_size):
    u_avg = np.reshape(u_avgs[i], (u_dim, 1)) 
    v_avgT = np.reshape(v_avgs[i], (1, v_dim)) 
    uvT_avg = u_avg @ v_avgT
    uvT_avgs = np.reshape(np.append(uvT_avgs, uvT_avg), (-1, 10, 10))   
    


#######################################################################
# PERFORMANCE EVALUATION: MSE 
#######################################################################

MSE = np_norm(uvT_gts - uvT_avgs)**2 / (n**2)
print(MSE) 

#######################################################################
# SIMULATIONS 
#######################################################################

