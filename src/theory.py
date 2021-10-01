from sympy.parsing.mathematica import mathematica
from sympy import *

import numpy as np

def theo_MMSE_fct(snr_gt, alpha_u, alpha_v, u_i_var_gt, u_i_var, v_i_var_gt, v_i_var):
    MSE_vect_pred_numerator = ((snr_gt**2) * alpha_u * alpha_v * (u_i_var_gt**2) * (v_i_var_gt**2) - 1.0)
    
    MSE_u_pred = MSE_vect_pred_numerator / ((snr_gt * alpha_u * v_i_var_gt) * (1.0 + snr_gt * alpha_v * v_i_var_gt * u_i_var_gt))
    MSE_v_pred = MSE_vect_pred_numerator / ((snr_gt * alpha_v * u_i_var_gt) * (1.0 + snr_gt * alpha_u * v_i_var_gt * u_i_var_gt))
    
    #return alpha_v * alpha_u * (u_i_var * v_i_var - MSE_u_pred * MSE_v_pred)
    return u_i_var * v_i_var - MSE_u_pred * MSE_v_pred

def theoretical_MMSE_function(alpha_u, alpha_v, u_i_var_gt, u_i_var, v_i_var_gt, v_i_var):
    def f(snr_gt_space):
        return np.piecewise(
            snr_gt_space,
            [snr_gt_space < 1, snr_gt_space >= 1],
            [lambda snr_gt: u_i_var_gt * v_i_var_gt, lambda snr_gt: theo_MMSE_fct(snr_gt, alpha_u, alpha_v, u_i_var_gt, u_i_var, v_i_var_gt, v_i_var)]
            )
    return f

a, L, J, s, t = var('a L J s t')
MMSE_sig_prod_expr = mathematica('s^2+2 a ((1/(2 a))Sqrt[L/J] (2-Sqrt[L/J]) (1/L-a/(L^2 s^2)+(1+L s^2)/L+((-1+a) s^2)/(a+L s^2)+(a+L s^2)/L-((1+L s^2) (a+L s^2))/(L^2 s^2)+((-1+a) J ((4 a (1+L s^2))/(J L t^2)+(4 a (a+L s^2))/(J L t^2)-(4 a (1+L s^2) (a+L s^2))/(J L^2 s^2 t^2)) t^2)/(2 Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)] (-2 a+J (1-a+Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))-((4 (1+L s^2) (a+L s^2) ((1+L s^2)/L+(a+L s^2)/L-((1+L s^2) (a+L s^2))/(L^2 s^2)-((4 a (1+L s^2))/(J L t^2)+(4 a (a+L s^2))/(J L t^2)-(4 a (1+L s^2) (a+L s^2))/(J L^2 s^2 t^2))/(2 Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)])))/(L s^2)+(4 (1+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L t^2)+(4 (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L t^2)-(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L^2 s^2 t^2))/(2 Sqrt[(-1+a)^2+(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L s^2 t^2)])+((1-a) ((4 (1+L s^2) (a+L s^2) ((1+L s^2)/L+(a+L s^2)/L-((1+L s^2) (a+L s^2))/(L^2 s^2)-((4 a (1+L s^2))/(J L t^2)+(4 a (a+L s^2))/(J L t^2)-(4 a (1+L s^2) (a+L s^2))/(J L^2 s^2 t^2))/(2 Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)])))/(L s^2)+(4 (1+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L t^2)+(4 (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L t^2)-(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L^2 s^2 t^2)))/(2 Sqrt[(-1+a)^2+(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L s^2 t^2)] (1-a+Sqrt[(-1+a)^2+(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L s^2 t^2)])))+(1/(2 a))(1/J+a/(J^2 t^2)+((-1+a) (-((2 a (1+L s^2) (a+L s^2))/(J L s^2 Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]))+(1-a+Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(-2 a+J (1-a+Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2)-((4 (1+L s^2) (a+L s^2) ((2 a (1+L s^2) (a+L s^2))/(J L s^2 Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)])+(((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L s^2 t^2)-(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J^2 L s^2 t^2))/(2 Sqrt[(-1+a)^2+(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L s^2 t^2)])+((1-a) ((4 (1+L s^2) (a+L s^2) ((2 a (1+L s^2) (a+L s^2))/(J L s^2 Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)])+(((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L s^2 t^2)-(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J^2 L s^2 t^2)))/(2 Sqrt[(-1+a)^2+(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L s^2 t^2)] (1-a+Sqrt[(-1+a)^2+(4 (1+L s^2) (a+L s^2) (a+J (((1+L s^2) (a+L s^2))/(L s^2)-Sqrt[(-1+a)^2+(4 a (1+L s^2) (a+L s^2))/(J L s^2 t^2)]) t^2))/(J L s^2 t^2)]))))')

def theoretical_MMSE_fct_of_sigma_product(alpha_v, snr_gt, snr, u_i_std_gt, v_i_std_gt):
    def f(u_i_std_times_v_i_std_space):
        L_val = snr_gt
        J_val = snr
        a_val = 1.0/alpha_v ## assuming alpha_u = 1, alpha_v = 1/alpha on the paper from Farzad
        s_val = u_i_std_gt * v_i_std_gt
        t_val_space = u_i_std_times_v_i_std_space
        
        return np.piecewise(
            t_val_space,
            [L_val * J_val * (s_val**2) * (t_val_space**2) > a_val, L_val * J_val * (s_val**2) * (t_val_space**2) <= a_val],
            [lambda t_val: lambdify([t], MMSE_sig_prod_expr.subs([(L, L_val),(J, J_val),(a, a_val),(s, s_val)]))(t_val), 
             lambda t_val: s_val**2]
        )
        
    return f