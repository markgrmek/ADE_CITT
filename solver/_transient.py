import numpy as np
from scipy.integrate import quad

#==================================================================
#GENERAL
#==================================================================
def calc_beta(
            U: float,
            D: float,
            eigval: float|np.ndarray
            ) -> float|np.ndarray:
        
        return np.sqrt(4*D*eigval**2 - U**2)/(2*D)

def calc_wghtm(
        xm_0: np.ndarray,
        Um: np.ndarray,
        Dm: np.ndarray,
        km: np.ndarray
        ) -> np.ndarray:
   
    k_D = km/Dm
    U_D = Um/Dm

    wght1 = np.exp(U_D[0]*xm_0[1])
    fac = k_D[1:]/k_D[:-1]
    exp_term = np.exp((U_D[1:] - U_D[:-1]) *xm_0[1:])

    #all weights are a cumulative linear combination of the first one
    res = wght1 * np.cumprod(fac*exp_term)
    res = np.insert(res, 0, wght1)
    return res

#==================================================================
#COEFFICIENTS AND TRANSCENETIAL EQUATION
#==================================================================

def calc_Tht_Am_Bm(
        x_int: float, 
        ThtA_prev: float|np.ndarray,
        ThtB_prev: float|np.ndarray, 
        U_prev: float,
        D_prev: float,
        k_prev: float,
        beta_prev: float,
        U_cur: float,
        D_cur: float,
        k_cur: float,
        beta_cur: float,
        ) -> float|np.ndarray:
    

    exp_fac = np.exp((0.5*U_prev/D_prev - 0.5*U_cur/D_cur)*x_int)
    interm = 0.5*U_prev*k_prev/D_prev - 0.5*U_cur*k_cur/D_cur

    A1_1 = np.sin(beta_cur*x_int)*np.sin(beta_prev*x_int)
    A1_2 = np.cos(beta_cur*x_int)/(beta_cur*k_cur)*(interm*np.sin(beta_prev*x_int) + beta_prev*k_prev*np.cos(beta_prev*x_int))

    A2_1 = np.sin(beta_cur*x_int)*np.cos(beta_prev*x_int)
    A2_2 = np.cos(beta_cur*x_int)/(beta_cur*k_cur)*(interm*np.cos(beta_prev*x_int) - beta_prev*k_prev*np.sin(beta_prev*x_int))

    ThtAm = (ThtA_prev*(A1_1 + A1_2) + ThtB_prev*(A2_1+A2_2))*exp_fac
    
    interm = 0.5*U_cur*k_cur/D_cur - 0.5*U_prev*k_prev/D_prev

    B1_1 = np.cos(beta_cur*x_int)*np.sin(beta_prev*x_int)
    B1_2 = np.sin(beta_cur*x_int)/(beta_cur*k_cur)*(interm*np.sin(beta_prev*x_int) - beta_prev*k_prev*np.cos(beta_prev*x_int))

    B2_1 = np.cos(beta_cur*x_int)*np.cos(beta_prev*x_int)
    B2_2 = np.sin(beta_cur*x_int)/(beta_cur*k_cur)*(interm*np.cos(beta_prev*x_int) + beta_prev*k_prev*np.sin(beta_prev*x_int))

    ThtBm = (ThtA_prev*(B1_1 + B1_2) + ThtB_prev*(B2_1+B2_2))*exp_fac

    return ThtAm, ThtBm

def calc_trans_eq(
          ThtAM: float|np.ndarray,
          ThtBM: float|np.ndarray,
          DM: float,
          UM: float,
          betaM: float|np.ndarray,
          alphaM1: float,
          alphaM2: float,
          xM: float = 1.0):
     
     term1 = alphaM1*np.sin(betaM*xM) + alphaM2*(0.5*UM*np.sin(betaM*xM)/DM + betaM*np.cos(betaM*xM))
     term2 = alphaM1*np.cos(betaM*xM) + alphaM2*(0.5*UM*np.cos(betaM*xM)/DM - betaM*np.sin(betaM*xM))

     return ThtAM*term1 + ThtBM*term2

#==================================================================
#CLOSED FORM INTEGRAL TRANSFORMS
#==================================================================

def calc_ImiPsimi(
        x: np.ndarray,
        ThtAmi: np.ndarray,
        ThtBmi: np.ndarray,
        betami: np.ndarray,
        wghtm: np.ndarray
        ) -> np.ndarray:
    
    fac = wghtm/(4*betami)

    N1 = ThtAmi**2*(2*betami*x - np.sin(2*betami*x))
    N2 = ThtBmi**2*(2*betami*x + np.sin(2*betami*x))
    N3 = 4*ThtAmi*ThtBmi*np.sin(betami*x)**2

    return fac*(N1+N2+N3)

def calc_ImiPhiGammi(
        x: np.ndarray,
        PhiGamAm: np.ndarray,
        PhiGamBm: np.ndarray,
        ThtAmi: np.ndarray,
        ThtBmi: np.ndarray,
        betami: np.ndarray,
        wghtm: np.ndarray,
        Um: np.ndarray,
        Dm: np.ndarray
        ) -> np.ndarray:
     
    fac = (2*wghtm*Dm)/(4*Dm**2*betami**2 + Um**2)

    PhiGam1 = ThtBmi*(2*Dm*betami*np.sin(betami*x) + Um*np.cos(betami*x))
    PhiGam2 = ThtAmi*(Um*np.sin(betami*x) - 2*Dm*betami*np.cos(betami*x))
    PhiGam3 = ThtBmi*(2*Dm*betami*np.sin(betami*x) - Um*np.cos(betami*x))
    PhiGam4 = ThtAmi*(Um*np.sin(betami*x) + 2*Dm*betami*np.cos(betami*x))

    term1 = PhiGamAm*(PhiGam1 + PhiGam2)*np.exp((Um*x)/(2*Dm))
    term2 = PhiGamBm*(PhiGam3 - PhiGam4)*np.exp(-(Um*x)/(2*Dm))

    return fac*(term1+term2)

def calc_ImiFm(
        x: np.ndarray,
        am: np.ndarray,
        bm: np.ndarray,
        ThtAmi: np.ndarray,
        ThtBmi: np.ndarray,
        betami: np.ndarray,
        wghtm: np.ndarray,
        Um: np.ndarray,
        Dm: np.ndarray
        ) -> np.ndarray:
     
    denom = 4*Dm**2*betami**2 + Um**2
    fac = 2*Dm*wghtm*np.exp(-(Um*x)/(2*Dm))/denom

    F1 = ThtBmi*(2*Dm*betami*np.sin(betami*x) - Um*np.cos(betami*x))
    F2 = ThtAmi*(2*Dm*betami*np.cos(betami*x) + Um*np.sin(betami*x))

    F3_1 = 8*Dm**3*betami**2*(betami*x*np.sin(betami*x) + np.cos(betami*x))
    F3_2 = 4*Dm**2*Um*betami*(2*np.sin(betami*x) - betami*x*np.cos(betami*x))
    F3_3 = 2*Dm*Um**2*(betami*x*np.sin(betami*x) - np.cos(betami*x))
    F3_4 = Um**3*x*np.cos(betami*x)

    F3 = F3_1 + F3_2 + F3_3 - F3_4

    F4_1 = 8*Dm**3*betami**2*(betami*x*np.cos(betami*x) - np.sin(betami*x))
    F4_2 = 4*Dm**2*Um*betami*(2*np.cos(betami*x) + betami*x*np.sin(betami*x))
    F4_3 = 2*Dm*Um**2*(betami*x*np.cos(betami*x) + np.sin(betami*x))
    F4_4 = Um**3*x*np.sin(betami*x)

    F4 = F4_1 + F4_2 + F4_3 + F4_4

    term1 = am*(F1 - F2)
    term2 = bm*(F3 - F4)/denom

    return fac*(term1 + term2)

#==================================================================
#ORTHOGONALITY CHECKER
#==================================================================
def calc_Nmij(
        ThtAmi: np.ndarray,
        ThtBmi: np.ndarray,
        betami: np.ndarray,
        wghtm: float,
        x_start: float, 
        x_end: float
        ) -> np.ndarray:
    
    num_norms = np.ndarray(shape=(len(betami), len(betami)))

    for i, beta_i in enumerate(betami):
        A_i = ThtAmi[i]
        B_i = ThtBmi[i]
        for j, beta_j in enumerate(betami):
            A_j = ThtAmi[j]
            B_j = ThtBmi[j]

            def integrand(x):
                psi_i = A_i*np.sin(x*beta_i) + B_i*np.cos(x*beta_i)
                psi_j = A_j*np.sin(x*beta_j) + B_j*np.cos(x*beta_j)
                return psi_i*psi_j
            
            result, er = quad(integrand, x_start, x_end, limit=200)

            num_norms[i][j] = result
    
    return wghtm*num_norms