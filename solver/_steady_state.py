import numpy as np

#=================================================================
#GENERAL
#=================================================================
def calc_sm(
        xm_0: np.ndarray,
        Um: np.ndarray,
        Dm: np.ndarray
        ) -> np.ndarray:
    
    U_D = Um/Dm
    sm = np.cumsum((U_D[:-1] - U_D[1:])*xm_0[1:])
    return np.insert(sm, 0, 0)

def calc_ACm(
        sm: np.ndarray,
        Um: np.ndarray,
        Dm: np.ndarray,
        km: np.ndarray
    ) -> np.ndarray:

    Uk_D = Um*km/Dm
    return Uk_D[0]*np.exp(sm)/Uk_D

def calc_BCm(
        xm_1: np.ndarray,
        sm: np.ndarray,
        Um: np.ndarray,
        Dm: np.ndarray,
        km: np.ndarray
    ) -> np.ndarray:

    Uk_D = Um*km/Dm
    U_D = Um/Dm

    diff_term = (Uk_D[:-1]**-1 - Uk_D[1:]**-1)
    exp_term = np.exp(sm[:-1] + U_D[:-1]*xm_1[:-1])
    BCm = np.cumsum(diff_term*exp_term)
    BCm = np.insert(BCm, 0, 0)
    return Uk_D[0]*BCm

def calc_Phi_Gam_Am(
        A1: float,
        ACm: np.ndarray
        ) -> np.ndarray:
    
    return A1*ACm

def calc_Phi_Gam_Bm(
        A1: float,
        B1: float,
        BCm: np.ndarray
        ) -> np.ndarray:
    
    return A1*BCm + B1

#=================================================================
#Phi (topBC = 1, bottomBC = 0)
#=================================================================
def calc_Phi_A1(
          alphaM1: float,
          alphaM2: float,
          DM: float,
          UM: float,
          ACM: float,
          BCM: float
          ) -> float:
     
    num = -alphaM1*DM
    exp_term = np.exp(UM/DM)
    denom = alphaM2*ACM*UM*exp_term + alphaM1*DM*(ACM*exp_term + BCM -1)

    return num/denom

def calc_Phi_B1(
          PhiA1: float
          ) -> float:
    
    return 1 - PhiA1

#=================================================================
#Gamma (topBC = 0, bottomBC = 1)
#=================================================================
def calc_Gam_A1(
          alphaM1: float,
          alphaM2: float,
          DM: float,
          UM: float,
          ACM: float,
          BCM: float
          ) -> float:
    
    num = DM
    exp_term = np.exp(UM/DM)
    denom = alphaM2*ACM*UM*exp_term + alphaM1*DM*(ACM*exp_term + BCM -1)

    return num/denom

def calc_Gam_B1(
          GamA1: float
          ) -> float:
    return - GamA1
