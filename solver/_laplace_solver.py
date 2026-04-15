import numpy as np

class LaplaceSolver:
    def __init__(
            self,
            U: float,
            D: float,
            T0: float,
            T1: float
            ) -> None:
        """
        
        Laplace solver for the ADE in a single homogeneous layer, with
        zero IC (T(x,0) = 0), zero Dirichlet BC on the outlet x=1 (T(1,t) =0)
        and a linearly increasing temperature on the linet x=0 (T(0,t) = T_0 + T_1*t).
        The solver is assumed to be nondimensionalized. Best is to nondimensonalize
        the parameters using the multilayer solver and plug them into this solver.

        Args:
            U (float): Nondimensional velocity of the thermal plume
            D (float): Nondimensional diffusivity
            T0 (float): Nondimensional intitial temperature at the inlet x=0
            T1 (float): Nondimensional rate of linear temperature change at the inlet x=0
        """        
        
        self.U, self.D = U, D
        self.T0, self.T1 = T0, T1

        #derived
        self.gam = 0.5*U/D

    def calcFinalSol(
            self,
            x_array: np.ndarray,
            t_array: np.ndarray,
            N_terms: int
            ) -> np.ndarray:

        x = x_array.reshape(-1,1)
        _j = np.arange(1, N_terms+1, 1).reshape(1,-1)

        _sj = self.gam**2 + _j**2*np.pi**2

        exp_term = np.exp(self.gam*x_array)
        sinh_term = np.sinh(self.gam*(1-x_array))/np.sinh(self.gam)

        sol_array = np.ndarray(shape=(t_array.shape[0], x_array.shape[0]))
        for idx, t in enumerate(t_array):
            #constant BC part
            sum1 = _j*np.sin(_j*np.pi*x)*np.exp(-self.D*_sj*t)/_sj
            sum1 = 2*np.pi*np.sum(sum1, axis=1)
            term1 = self.T0*(sinh_term - sum1)*exp_term

            #linearly increasing BC part
            sum2 = _j*np.sin(_j*np.pi*x)*(1-np.exp(-self.D*_sj*t))/(_sj**2)
            sum2 = 2*np.pi*np.sum(sum2, axis=1)/self.D
            term2 = self.T1*(sinh_term*t - sum2)*exp_term

            sol_array[idx] = term1 + term2

        return sol_array