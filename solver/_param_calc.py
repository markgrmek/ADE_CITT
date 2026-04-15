import numpy as np

def calc_D(
        lamb: float|np.ndarray,
        C: float|np.ndarray
        ) -> float|np.ndarray:
    """Calculate the diffusivity/ies D (m/s^2)

    Args:
        lamb (float): Thermal conductivity of the soil/water matrix (J/msK)
        C (float): Volumetric heat capacity of the soil/water matrix (J/m^3K)

    Returns:
        float: D = lamb/C (m/s^2)
    """
    
    C = np.array(C)
    lamb = np.array(lamb)

    return lamb/C

def calc_U(
        q: float,
        C: float|np.ndarray,
        Cw: float
        ) -> float|np.ndarray:
    
    """Calculate the thermal velocity/ies U (m/s)

    Args:
        q (float): Vertical flow rate (m/s)
        C (float | np.ndarray): Volumetric heat capacity of the soil/water matrix (J/m^3K)

    Returns:
        float|np.ndarray: U = q*Cw/C (m/s)
    """    
    
    C = np.array(C)

    return q*Cw/C


#============================================
#EQUIVALENT PARAMETERS
#============================================
def calcParamEq(
        param: np.ndarray,
        xm_0: np.ndarray,
        xm_1: np.ndarray
        ) -> float:
    """Calculate the equivalent (harmonic mean) parameter of choice

    Args:
        param (np.ndarray): parameter values in each layer
        xm_0 (np.ndarray): layer starts
        xm_1 (np.ndarray): layer ends

    Returns:
        float: single param_eq parameter
    """    
    return xm_1[-1]/np.sum((xm_1-xm_0)/param)
