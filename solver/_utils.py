import numpy as np
from scipy.interpolate import interp1d
from utils.metrics import RMSE

#==========================================
#INITIAL CONDITIONS
#==========================================
def calcInterTemps(
        T_mes: np.ndarray,
        x_mes: np.ndarray,
        xm_1: np.ndarray,
        T_surface: float
        ) -> tuple[np.ndarray, float]:
    """Calculate the T at the interfaces x_m; m=0,1, ... M from the 
    initial GWT profile using linear interpolation

    Args:
        T_mes (np.ndarray): measured GWT at t=0
        x_mes (np.ndarray): x of the measured GWT at t=0
        xm_1 (np.ndarray): layer interfaces x_m for m=1,2, ... M
        T_surface (float): GST on the surface at t=0, i.e. v_0(0)

    Returns:
        tuple[np.ndarray, float]: The array of predicted IC GWT at x_m for m=0,1, ... M and the RMSE of the IC GWT
    """    

    T_mes = np.array(T_mes)
    x_mes = np.array(x_mes)
    xm_1 = np.array(xm_1)

    if not x_mes[0] == 0: #in case the IC dont start at x=0 - ensure compliance between BC and IC
        T_mes = np.insert(T_mes, 0, T_surface)
        x_mes = np.insert(x_mes, 0, 0)

    xm_1 = np.insert(xm_1, 0, 0) #x1 by default has no x0=0
    
    inter = interp1d(x_mes, T_mes, fill_value='extrapolate') #interpolator
    fitted = inter(xm_1) #T at interfaces x_m calcualted from the measured (x,T) points

    #ERROR ESTIMATION--------------------------------------------
    er_inter = interp1d(xm_1, fitted, fill_value='extrapolate') #create second interpolator from the IC data
    er_fitted = er_inter(x_mes) #calculate the values of the IC at the x_measured points
    error = RMSE(T_mes, er_fitted) #get the error between IC and T_measured

    print(f'RMSE of the fitted IC: {error:.2e}')

    return fitted

