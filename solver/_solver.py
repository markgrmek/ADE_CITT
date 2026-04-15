import numpy as np
from pandas import Series
import warnings

from ._eigenvalue_finder import *
from ._steady_state import *
from ._transient import *
from ._param_calc import *
from ._plotting import *
from ._utils import *

class Solver:
    def __init__(
            self,
            xm_0: list|np.ndarray|Series,
            xm_1: list|np.ndarray|Series,
            q: float,
            lambm: list|np.ndarray|Series,
            Cm: list|np.ndarray|Series,
            km: list|np.ndarray|Series,
            surface_temps: list|np.ndarray|Series,
            surface_times: list|np.ndarray|Series,
            alphaM1: float,
            alphaM2: float,
            interface_temps: list|np.ndarray|Series, #temeperatures at x_m including x_0 at t==0
            T_ref: float = 1.0,
            C_in_MJ: bool = True
        ) -> None:
        """

        Multilayer CITT solver for the ADE in a multi-layer medium

        Args:
            x0 (list | np.ndarray | Series): starting points of each layer (m)
            x1 (list | np.ndarray | Series): end points of each layer (m)
            q (float): vertical hydraulic flow rate (m/s) - asumed constant across all layers
            lamb (list | np.ndarray | Series): thermal conductivity of each layer (J/msK)
            C (list | np.ndarray | Series): volumetric heat capacities of each layer (MJ/m^3K) or (J/m^3K)
            k (list | np.ndarray | Series): generic coefficients to preserve flux
            surface_temps (list | np.ndarray | Series): surface temperatures at time t=t_j (C)
            surface_times (list | np.ndarray | Series): times t=t_j, j=1,2,... of surface temperatures (s)
            alpha1 (float): Dirichlet BC control coefficient on the outlet x=x_M
            alpha2 (float): Neumann BC control coefficient on the outlet x=x_M
            interface_temps (list | np.ndarray | Series): temperatures at t=0 on the layer interfaces x=x_i for i=1,2...M
            T_ref (float): Reference temperatrue used for non-dimensionalization (C)
            C_in_MJ (bool, optional): True in case volumetric heat capacity is in MJ/m^3, false for J/m^3. Defaults to True.

        Raises:
            ValueError: in case the lenghts of x0, x1, lamb, C and k arrays are not all M
            ValueError: in case the length of interface temperatures array is not M+1 (must include also x_0)
        """        

        if not all([len(arr) == len(xm_0) for arr in (xm_1, lambm, Cm, km)]):
            raise ValueError('Layer arrays x0, x1, k, lamb and C must be of the same lenght')

        if not len(interface_temps) == len(xm_0)+1:
            raise ValueError('Interface temepratures must be provided for all interfaces x_m including the endpoints x_0 and x_M')

        #Misc
        self.Cw: float = 4.18*1e6 #volumetric heat capacity of water in (J/m^3K) - assumed constant
        self.T_ref = T_ref
        self.sec_per_year = 365.25*24*60*60
        self.sec_per_month = self.sec_per_year/12
        self.nondim: bool = False #indicator that the system was nondimensionalized

        #Layers
        self.xm_0 = np.array(xm_0)
        self.xm_1 = np.array(xm_1)
        self.km = np.array(km)
        self.Cm = np.array(Cm)*1e6 if C_in_MJ else np.array(Cm) #convert to (J/m^3K)
        self.lambm = np.array(lambm)
        self.q = q

            #derived parameters
        self.Um = calc_U(q, self.Cm, self.Cw)
        self.Dm = calc_D(lambm, self.Cm)
        
        #BCs
        #inlet
        self.surface_temps = np.array(surface_temps, dtype=float)
        self.surface_times = np.array(surface_times, dtype=float)
        self.surface_t0 = self.surface_times[0]
        self.surface_times -= self.surface_t0
        self.surface_dT_dt = np.diff(self.surface_temps)/np.diff(self.surface_times)

        #outlet
        self.alphaM1 = alphaM1 #bottom BC controls
        self.alphaM2 = alphaM2

        #ICs
        interface_temps = np.array(interface_temps)
        self.IC_bm = (interface_temps[1:]-interface_temps[:-1])/(self.xm_1 - self.xm_0)
        self.IC_am = interface_temps[:-1] - self.IC_bm*self.xm_0

    #======================================================
    #GENERAL SOLVER PARAMETERS
    #======================================================
    def nonDimensionalize(self) -> None:
        """Nondimensionalize the system
        """        
        self.nondim = True

        #set x0 to 0
        self.x_shft = self.xm_0.min() #store for later use
        self.xm_0 -= self.x_shft
        self.xm_1 -= self.x_shft

        #store parameters for later use
        self.xM = self.xm_1[-1]
        self.kM = self.km[-1]
        self.DM = self.Dm[-1]
        self.UM = self.Um[-1]
        self.CM = self.Cm[-1]

        self.Cm /= self.CM
        self.lambm /= (self.q*self.Cw*self.xM)
        self.q /= self.q #nondim flow rate is 1
        self.xm_0 /=  self.xM
        self.xm_1 /=  self.xM
        self.Dm /= (self.UM*self.xM)
        self.Um /= self.UM
        self.km /= self.kM

        #BCs
        self.surface_temps /= self.T_ref #C/C == 1
        self.surface_times *= (self.UM/self.xM) #m/s * m == 1/s => s/s == 1
        self.alphaM2 /= self.xM
        self.surface_dT_dt = np.diff(self.surface_temps)/np.diff(self.surface_times)

        #ICs
        self.IC_am /= self.T_ref
        self.IC_bm *= self.xM/self.T_ref

    #helper functions to nondimensinoalize external arrays
    def nonDim_x(
            self, 
            x: np.ndarray
            ) -> np.ndarray:
        """Nondimensionalize a desired dimensional x array

        Args:
            x (np.ndarray): dimensional x array

        Returns:
            np.ndarray: nondimensional x array
        """        
        return x/self.xM
    
    def nonDim_t(
            self, 
            t: np.ndarray|float
            ) -> np.ndarray|float:
        """Nondimensionalize a desired dimensional t array

        Args:
            t (np.ndarray): dimensional t array

        Returns:
            np.ndarray: nondimensional t array
        """        
        return t*self.UM/self.xM

    #======================================================
    #STEADY STATE SOLUTIONS
    #=====================================================
    def calcStdSolParams(self) -> None:
        """Calculate the steady state solution parameters
        """        
        #general coefficients
        sm = calc_sm(self.xm_0, self.Um, self.Dm)
        ACm = calc_ACm(sm, self.Um, self.Dm, self.km)
        BCm = calc_BCm(self.xm_1, sm, self.Um, self.Dm, self.km)

        #Phi solution
        PhiA1 = calc_Phi_A1(self.alphaM1, self.alphaM2, self.Dm[-1], self.Um[-1], ACm[-1], BCm[-1])
        PhiB1 = calc_Phi_B1(PhiA1)

        self.PhiAm = calc_Phi_Gam_Am(PhiA1, ACm)
        self.PhiBm = calc_Phi_Gam_Bm(PhiA1, PhiB1, BCm)

        #Gamma solution
        GamA1 = calc_Gam_A1(self.alphaM1, self.alphaM2, self.Dm[-1], self.Um[-1], ACm[-1], BCm[-1])
        GamB1 = calc_Gam_B1(GamA1)

        self.GamAm = calc_Phi_Gam_Am(GamA1, ACm)
        self.GamBm = calc_Phi_Gam_Bm(GamA1, GamB1, BCm)

    def __sol_Phi(
            self,
            x: np.ndarray,
            region_idxs: np.ndarray
            ) -> np.ndarray:
        
        U = self.Um[region_idxs]
        D = self.Dm[region_idxs]
        A = self.PhiAm[region_idxs]
        B = self.PhiBm[region_idxs]

        return A*np.exp(U*x/D) + B
    
    def __sol_Gam(self,
            x: np.ndarray,
            region_idxs: np.ndarray
            ) -> np.ndarray:
        
        U = self.Um[region_idxs]
        D = self.Dm[region_idxs]
        A = self.GamAm[region_idxs]
        B = self.GamBm[region_idxs]

        return A*np.exp(U*x/D) + B
            
    #======================================================
    #TRANSIENT SOLUTION
    #=====================================================      
    def findEigenvalues(
            self,
            num_eigvals: int,
            step: float = 0.5,
            tol: float = 1e-20,
            show_plot: bool = True,
            fig_width: float = 12.0,
            savefig: bool = False
            ) -> None:
        """Find the eigenvalues of the EVP

        Args:
            num_eigvals (int): Number of desired eigenvalues
            step (float, optional): Step in the eigenvalue direction - used for bracketing. Defaults to 0.5.
            tol (float, optional): Eigenvalue precision tolerance. Defaults to 1e-20.
            show_plot (bool, optional): Show the trascnendential equation plot. Defaults to True.
        """        
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning) #this function is bound to return when encountering negative roots
            
            self.eigenvals = find_eigvals(
                trans_eq=self.__transendential_eq,
                num_eigvals= num_eigvals,
                step=step,
                tol=tol)

        if show_plot:
            #calculaiton-------------------------------------------
            min_eig, max_eig = self.eigenvals.min(), self.eigenvals.max()
            x_array = np.linspace(min_eig, max_eig, int(max_eig//step))
            f_array = self.__transendential_eq(x_array)

            #plotting----------------------------------------------
            plotTranscendential(x_array, f_array, self.eigenvals, fig_width, savefig)

    def calcTransSolParams(self) -> None:
        """Calculate the transient solution components - components transformed with the integral operator 
        """        
        #general parameters-------------------------------------
        self.__calc_trans_sol_coefficients()
        self.wghtm = calc_wghtm(self.xm_0, self.Um, self.Dm, self.km)

        #reshaping arrays for full vectorization
        _Um = self.Um.reshape(-1,1)
        _Dm = self.Dm.reshape(-1,1)
        _wghtm = self.wghtm.reshape(-1,1)
        _xm_0 = self.xm_0.reshape(-1,1)
        _xm_1 = self.xm_1.reshape(-1,1)
        
        #Norm------------------------------------------
        args1 = (_xm_1, self.ThtAmi, self.ThtBmi, self.betami, _wghtm)
        args2 = (_xm_0, self.ThtAmi, self.ThtBmi, self.betami, _wghtm)
        self.Ni = np.sum(calc_ImiPsimi(*args1) - calc_ImiPsimi(*args2), axis=0) #sum over layers

        #Transformed Phi_m(x)-------------------------
        _PhiAm = self.PhiAm.reshape(-1,1)
        _PhiBm = self.PhiBm.reshape(-1,1)
        args1 = (_xm_1, _PhiAm, _PhiBm, self.ThtAmi, self.ThtBmi, self.betami, _wghtm, _Um, _Dm)
        args2 = (_xm_0, _PhiAm, _PhiBm, self.ThtAmi, self.ThtBmi, self.betami, _wghtm, _Um, _Dm)
        self.ImiPhim = np.sum(calc_ImiPhiGammi(*args1) - calc_ImiPhiGammi(*args2), axis=0) #sum over layers

        #Trnasformed Gamma_m(x)-----------------------
        _GamAm = self.GamAm.reshape(-1,1)
        _GamBm = self.GamBm.reshape(-1,1)
        args1 = (_xm_1, _GamAm, _GamBm, self.ThtAmi, self.ThtBmi, self.betami, _wghtm, _Um, _Dm)
        args2 = (_xm_0, _GamAm, _GamBm, self.ThtAmi, self.ThtBmi, self.betami, _wghtm, _Um, _Dm)
        self.ImiGamm = np.sum(calc_ImiPhiGammi(*args1) - calc_ImiPhiGammi(*args2), axis=0) #sum over layers

        #Transformed IC F_m(x)-----------------------
        _am = self.IC_am.reshape(-1,1)
        _bm = self.IC_bm.reshape(-1,1)
        args1 = (_xm_1, _am, _bm, self.ThtAmi, self.ThtBmi, self.betami, _wghtm, _Um, _Dm)
        args2 = (_xm_0, _am, _bm, self.ThtAmi, self.ThtBmi, self.betami, _wghtm, _Um, _Dm)
        self.ImiFm = np.sum(calc_ImiFm(*args1) - calc_ImiFm(*args2), axis=0) #sum over layers

    #TRANSENDENTIAL EQUATION ---------------------------------------------------
    def __transendential_eq(
            self,
            eigval: float
            ) -> float:
        
        #This function gets used only once - when determining the eigenvalues
        #p == previous == m-1; c == current == m

        ThtA_p, ThtB_p = 1, 0
        U_p, D_p, k_p = self.Um[0], self.Dm[0], self.km[0]
        beta_p = calc_beta(U_p, D_p, eigval)

        for idx, x_int in enumerate(self.xm_0[1:], 1):
            U_c, D_c, k_c = self.Um[idx], self.Dm[idx], self.km[idx]
            beta_c = calc_beta(self.Um[idx], self.Dm[idx], eigval)
            
            #calulcate ThtAm and ThtBm
            ThtA_c, ThtB_c = calc_Tht_Am_Bm(
                x_int,
                ThtA_p, ThtB_p, U_p, D_p, k_p, beta_p,
                U_c, D_c, k_c, beta_c
                )

            #update the previous variables
            ThtA_p, ThtB_p, beta_p = ThtA_c, ThtB_c, beta_c
            U_p, D_p, k_p = U_c, D_c, k_c
            
        #cur_lay in last iteration is automatically last layer
        return calc_trans_eq(ThtA_c, ThtB_c, D_c, U_c, beta_c, self.alphaM1, self.alphaM2)

    #COEFFICIENT CALCULAITON------------------------------------------------------
    def __calc_trans_sol_coefficients(self) -> None:
        #this function calculates the needed parameters when the eigenvalues have been determined
        #p == previous == m-1; c == current == m

        #calculate for the first layer seperatley
        ThtA_p, ThtB_p = np.ones_like(self.eigenvals), np.zeros_like(self.eigenvals)
        U_p, D_p, k_p = self.Um[0], self.Dm[0], self.km[0]
        beta_p = calc_beta(U_p, D_p, self.eigenvals)

        ThtAmi, ThtBmi, betami = [ThtA_p], [ThtB_p], [beta_p] #add the first layer manually

        for idx, x_int in enumerate(self.xm_0[1:], 1):
            U_c, D_c, k_c = self.Um[idx], self.Dm[idx], self.km[idx]
            beta_c = calc_beta(U_c, D_c, self.eigenvals)
            
            #calulcate ThtA and ThtB
            ThtA_c, ThtB_c = calc_Tht_Am_Bm(
                x_int,
                ThtA_p, ThtB_p, U_p, D_p, k_p, beta_p,
                U_c, D_c, k_c, beta_c
                )
            
            #append results
            betami.append(beta_c)
            ThtAmi.append(ThtA_c)
            ThtBmi.append(ThtB_c)
        
            #update the previous variables
            ThtA_p, ThtB_p, beta_p = ThtA_c, ThtB_c, beta_c
            U_p, D_p, k_p = U_c, D_c, k_c

        #ADD DATA TO DATAFRAME (stack to convert to 2D array)
        self.betami = np.stack(betami)
        self.ThtAmi = np.stack(ThtAmi)
        self.ThtBmi = np.stack(ThtBmi)

    def __sol_psi(
            self,
            x: np.ndarray,
            region_idxs: np.ndarray
            ) -> np.ndarray:
        
        x = x.reshape(-1,1)
        U = self.Um[region_idxs].reshape(-1,1)
        D = self.Dm[region_idxs].reshape(-1,1)
        beta = self.betami[region_idxs]
        A = self.ThtAmi[region_idxs]
        B = self.ThtBmi[region_idxs]

        return np.exp((U*x)/(2*D))*(A*np.sin(beta*x) + B*np.cos(beta*x))

    #======================================================
    #FULL SOLUTIONS
    #======================================================
    #SUB FUNCTION FOR COMPUTING SOLUTION AT A SINGLE TIME POINT-----------------------
    def __single_t_sol(
            self,
            t: float,
            ImiThtIC: np.ndarray,
            psi: np.ndarray,
            Phi: np.ndarray,
            Gam: np.ndarray
            ) -> np.ndarray:
        
        #IC term in the Theta solution -----------------------------------------
        init_term = np.exp(-self.eigenvals**2*t)*ImiThtIC*psi/self.Ni 

        # Duhamel integral--------------------------------------------------------
        time_filter = self.surface_times <= t
        used_times = self.surface_times[time_filter]
        used_temps = self.surface_temps[time_filter]
        used_dT_dt = self.surface_dT_dt[time_filter[:-1]]  # This is correct! dT_dt has length len(used_times)-1

        sum_0_to_J = np.zeros_like(self.eigenvals)
        for idx, (t0, t1) in enumerate(zip(used_times[:-1], used_times[1:])):
            sum_0_to_J += used_dT_dt[idx] * (
                np.exp(-self.eigenvals**2 * (t - t1)) - 
                np.exp(-self.eigenvals**2 * (t - t0))
            )

        # For J_term, we need the last dT_dt value (which corresponds to the interval [t_{J-1}, t_J])
        J_term = used_dT_dt[-1] * (1 - np.exp(-self.eigenvals**2 * (t - used_times[-1])))
        # if len(used_dT_dt) > 0:
        #     J_term = used_dT_dt[-1] * (1 - np.exp(-self.eigenvals**2 * (t - used_times[-1])))
        # else:
        #     J_term = 0  # Handle case with no time points

        duhamel_int = self.ImiPhim * psi * (sum_0_to_J + J_term) / (self.Ni * self.eigenvals**2)

        #Combined Theta solution----------------------------------------------------
        Tht = np.sum(init_term - duhamel_int, axis=1) #sum over all eigenvalues

        #combining the superposition-----------------------------------------------
        v0t = used_temps[-1] + used_dT_dt[-1]*(t-used_times[-1])
        vM = self.alphaM1*(self.IC_am[-1] + self.IC_bm[-1]) + self.alphaM2*(self.IC_bm[-1])

        return Tht + Phi*v0t + Gam*vM
    
    def calcFinalSol(
            self,
            x_array: np.ndarray,
            time: np.ndarray|float
            ) -> np.ndarray:
        
        """Calculate the solution for arbitrary x points for all provided times t

        Args:
            x_array (np.ndarray): array of x points
            time (np.ndarray|float): singlar time or array of times

        Returns:
            np.ndarray: Solution array. Returns a 1D array for if time is a float or a 2D array with dim (len(time),len(x_array) if time is a np.ndarray.
        """  

        #constant params---------------------------------------     
        v00 = self.surface_temps[0] #inlet BC at t==0 v_0(0)
        vM = self.alphaM1*(self.IC_am[-1] + self.IC_bm[-1]) + self.alphaM2*(self.IC_bm[-1]) #outlet BC for all t
        ImiThtIC = self.ImiFm - self.ImiPhim*v00 - self.ImiGamm*vM #normalized transformed initial conds  - I_ni Theta_n(x', 0)

        #seperate solutions depending only on x------------------------------
        region_idxs = np.searchsorted(self.xm_0, x_array, side='right') - 1
        Phi = self.__sol_Phi(x_array, region_idxs)
        Gam = self.__sol_Gam(x_array, region_idxs)
        psi = self.__sol_psi(x_array, region_idxs)

        #solution part--------------------------------------
        if isinstance(time, float):
            solution = self.__single_t_sol(time, ImiThtIC, psi, Phi, Gam)
        
        elif isinstance(time, np.ndarray):
            solution = np.ndarray((len(time), len(x_array))) #result array allocation
            for idx, t in enumerate(time):
                solution[idx] = self.__single_t_sol(t, ImiThtIC, psi, Phi, Gam)

        return solution

    #===============================================================================
    #PLOTTING
    #===============================================================================
    def plotSteadySol(
            self, 
            N_res_pnt: int = 100, 
            fig_width: float = 12.0, 
            savefig: bool = False,
            add_sinh_sol: bool = False
            ) -> None:
        """Plot the steady state solutions of the CITT

        Args:
            N_res_pnt (int, optional): Number of points in x. Defaults to 100.
            fig_width (float, optional): With of the figure. Defaults to 12.0.
            savefig (bool, optional): Save figure to a .png file. Defaults to False.
            add_sinh_sol (bool, optional): Add the Laplace steady state solution as a control. Defaults to False.
        """        
        
        #calculation----------------------------------
        x_array = np.linspace(self.xm_0[0], self.xm_1[-1], N_res_pnt)
        region_idxs = np.searchsorted(self.xm_0, x_array, side='right') - 1

        Phi = self.__sol_Phi(x_array, region_idxs)
        Gam = self.__sol_Gam(x_array, region_idxs)

        #sinh steady state soluton (from laplace)------
        #it is assumed that the
        sinh_term = None
        if add_sinh_sol:
            gam = (2*self.Dm[0])**-1
            sinh_term = np.sinh(gam*(1-x_array))*np.exp(gam*x_array)/np.sinh(gam)

        #plotting-------------------------------------
        plotSteadyState(
            x_array=x_array,
            Phi=Phi,
            Gam=Gam,
            xm_1=self.xm_1, 
            alphaM1=self.alphaM1, 
            alphaM2=self.alphaM2, 
            savefig=savefig,
            fig_width=fig_width, 
            sinh_sol=sinh_term
            )

    def plotOrthogonality(
            self,
            print_results: bool = True,
            max_ticks: int = 10,
            fig_width: float = 6.0,
            savefig: bool = False
            ) -> None:
        """Plot the orthogonality of the EVP

        Args:
            print_results (bool, optional): Print the a general summary of results. Defaults to True.
            max_ticks (int, optional): Maximum number of ticks on the x and y axis. Defaults to 10.
            fig_width (float, optional): Width of the figure. Defaults to 3.0.
            savefig (bool, optional): Save the figure to a .png file. Defaults to False.
        """    

        #calculation---------------------------------
        N_eigvals = len(self.eigenvals)
        Nij = np.zeros(shape=(N_eigvals, N_eigvals))
        for idx, betam in enumerate(self.betami):
            Nij += calc_Nmij(
                self.ThtAmi[idx], self.ThtBmi[idx], betam, 
                self.wghtm[idx], self.xm_0[idx], self.xm_1[idx]
            )
        
        #plotting---------------------------------
        plotFullNorm(
            Nij=Nij,
            print_results=print_results,
            max_ticks=max_ticks, 
            fig_width=fig_width,
            savefig=savefig
            )
    
    def plotConvergence(
            self,
            max_ticks: int = 10,
            fig_width: float = 12.0,
            savefig: bool = False
            ) -> None:
        """Plot the convergence of the main CITT solution components

        Args:
            max_ticks (int, optional): Maximum number of ticks on the x axis. Defaults to 10.
            fig_width (float, optional): Width of the figure. Defaults to 12.0.
            savefig (bool, optional): Save the figure to a .png file. Defaults to False.
        """        
        
        #plotting------------------------------
        plotConvergence(self.Ni, self.ImiPhim, self.ImiGamm, self.ImiFm, max_ticks, fig_width, savefig)

    def plotInitialConds(
            self,
            fig_width: float = 12.0,
            savefig: bool = False
            ) -> None:
        """Plot the piecewise-linear initial conditions

        Args:
            fig_width (float, optional): Width of the figure. Defaults to 12.0.
            savefig (bool, optional): Save the figure to a .png file. Defaults to False.
        """       

        #calculation---------------------------------------
        x_array = np.linspace(self.xm_0[0], self.xm_1[-1], 100)
        region_idxs = np.searchsorted(self.xm_0, x_array, side='right') - 1

        Fa = self.IC_am[region_idxs]
        Fb = self.IC_bm[region_idxs]

        Fm = Fa + Fb*x_array
        #plotting------------------------------------------
        plotIC(
            x_array=x_array, 
            Fm=Fm, 
            xm_1=self.xm_1, 
            fig_width=fig_width, 
            savefig=savefig
            )

    def plotInletBCs(
            self,
            fig_width: float = 12.0,
            savefig: bool = False
            ) -> None:
        """Plot the piecewise-linear initial conditions

        Args:
            fig_width (float, optional): Width of the figure. Defaults to 12.0.
            savefig (bool, optional): Save the figure to a .png file. Defaults to False.
        """    

        #plotting------------------------------
        plotInletBC(
            t_array=self.surface_times,
            temp_array=self.surface_temps,
            fig_width=fig_width,
            savefig=savefig
            )