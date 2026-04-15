import numpy as np
import matplotlib.pyplot as plt

#==================================================================
#MISC
#==================================================================
_PLTCONTEXT: dict = {
    'xtick.color': "#000000",
    'ytick.color': "#000000",
    'axes.edgecolor': "#000000",
    'axes.labelcolor': "#000000",
    'axes.titlecolor': "#000000",
    "axes.grid": False,
    "axes.grid.axis": "y",
    "grid.linestyle": ":",
    'ytick.major.size': 0,
    'xtick.major.size': 0,
    'axes.linewidth': 0.5,
    'lines.markersize': 4,
    'lines.linewidth': 0.7,
    'errorbar.capsize': 2,
    'font.family': 'sans-serif',  # or 'serif', 'monospace'
    'font.serif': 'Arial',
    'mathtext.fontset': 'dejavusans',
    'mathtext.rm': 'sansserif',
    'font.size': 10,  # Base font size
    'axes.titlesize': 10,      # Title font size
    'axes.labelsize': 10,      # Axis label font size
    'xtick.labelsize': 10,      # X-tick label font size
    'ytick.labelsize': 10,      # Y-tick label font size
    'legend.fontsize': 9,      # Legend font size
}

_KWARGS_LEGEND: dict = dict(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
_KWARGS_SAVEFIG: dict =  dict(bbox_inches="tight", dpi=300)
_KWARGS_INTERFACES: dict = dict(linestyle=':', color='gray', alpha=0.5)
_KWARGS_MEAS: dict = dict(linestyle='none', fillstyle='none', markersize=3)
_MARKERS: list[str] = ['o', 's', '^', 'D', 'v', '>', '<', 'p', 'h', 'H', '*', 'X', 'P', '8', 'd', 'o']

def _create_cmap(
          N_lines: int
          ) -> np.ndarray:
    
    cmap = plt.colormaps.get_cmap('viridis')
    return cmap(np.linspace(0, 0.7, N_lines)) #up to 0.7 the colors are still nice

def _sci_notation(
          x: float, 
          prec: int = 2
          ) -> str:
    
    if x == 0:
        return r"$0$"

    if abs(x - round(x)) < 1e-15:
        return rf"${int(round(x))}$"

    formatted = f"{x:.{prec}g}"
    formatted = formatted.replace('e', 'E')
    
    return rf"${formatted}$"

#==================================================================
#NON-CALCULABLE
#==================================================================
def plotIC(
        x_array: np.ndarray,
        Fm: np.ndarray,
        xm_1: np.ndarray,
        fig_width: float = 12.0,
        savefig: bool = False
        ) -> None:
    
    with plt.rc_context(_PLTCONTEXT):
        fig = plt.figure(figsize=(fig_width, 3))
        ax = fig.add_subplot(1, 1, 1)

        #plot lines
        ax.plot(x_array, Fm, color='#440154', label = r'$F_m(x)$')

        for i in xm_1[:-1]: 
            ax.axvline(i, **_KWARGS_INTERFACES) #plot interfaces

        #set limits
        ax.set_ylim(Fm.min(), Fm.max())
        ax.set_xlim(x_array.min(), x_array.max())

        #add texts
        ax.set_xlabel('x')
        ax.set_ylabel('T')
        ax.legend(**_KWARGS_LEGEND)

        if savefig: 
             fig.savefig('initial_cond.png', **_KWARGS_SAVEFIG)

        plt.show()

def plotInletBC(
        t_array: np.ndarray,
        temp_array: np.ndarray,
        fig_width: float,
        savefig: bool
        ) -> None:
    
    with plt.rc_context(_PLTCONTEXT):
        fig = plt.figure(figsize=(fig_width, 3))
        ax = fig.add_subplot(1, 1, 1)

        #plot lines
        ax.plot(t_array, temp_array, color='#440154', label = r'$v_0(t)$')

        #set limits
        ax.set_ylim(temp_array.min(), temp_array.max())
        ax.set_xlim(t_array.min(), t_array.max())

        #add labels
        ax.set_xlabel('t')
        ax.set_ylabel('T')
        ax.legend(**_KWARGS_LEGEND)

        if savefig: 
             fig.savefig('inlet_BC.png', **_KWARGS_SAVEFIG)
        
        plt.show()


#==================================================================
#STEADY STATE SOLUTIONS
#==================================================================
def plotSteadyState(
            x_array: np.ndarray,
            Phi: np.ndarray,
            Gam: np.ndarray,
            xm_1: np.ndarray,
            alphaM1: float,
            alphaM2: float,
            sinh_sol: None|np.ndarray,
            fig_width: float,
            savefig: bool,
            ) -> None:
        
        #BC checking------------------------------------------------------------
        #inlet
        Phi_in = Phi[0]
        Gam_in = Gam[0]

        #outlet
        dx = x_array[1]-x_array[0]
        Phi_out = alphaM1*Phi[-1] + alphaM2*np.gradient(Phi, dx, edge_order=2)[-1]

        Gam_out = alphaM1*Gam[-1] + alphaM2*np.gradient(Gam, dx, edge_order=2)[-1]

        cmap = _create_cmap(3 if sinh_sol is not None else 2)
        markevery = int(len(x_array)/10)

        x_min, x_max = x_array[0], x_array[-1]
        y_min, y_max = np.min((Phi.min(), Gam.min())), np.max((Phi.max(), Gam.max()))

        text = (
            rf"$\Phi_0$ = {_sci_notation(Phi_in, 1)}" 
            + "\n"  
            rf"$\alpha_M\Phi_M+ \alpha^*_M\Phi'_M$ = {_sci_notation(Phi_out, 1)}"
            + "\n \n" 
            + rf"$\Gamma_0$ = {_sci_notation(Gam_in, 1)}"
            + "\n"
            +  rf"$\alpha_M\Gamma_M+ \alpha^*_M\Gamma'_M$ = {_sci_notation(Gam_out, 1)}"
        )

        with plt.rc_context(_PLTCONTEXT):
            fig = plt.figure(figsize=(fig_width, fig_width/3))
            ax = fig.add_subplot(1, 1, 1)

            #plot lines----------------------------------------------------------
            ax.plot(x_array, Gam, color = cmap[0], label = r'$\Gamma_m(\tilde{x})$')
            ax.plot(x_array, Phi, color = cmap[1], label = r'$\Phi_m(\tilde{x})$')
            
            for i in xm_1[:-1]: 
                 ax.axvline(i, **_KWARGS_INTERFACES) #plot interfaces

            if sinh_sol is not None:
                 ax.plot(x_array, sinh_sol, color=cmap[2], marker=_MARKERS[2], markevery=(markevery//2, markevery), label=r'$\overline{\Phi}(\tilde{x})$', **_KWARGS_MEAS)

            #set limits-----------------------------------------------------------
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)

            #add texts--------------------------------------------------------------
            ax.text(x_max + 0.02*(x_max-x_min), y_min, text, ha='left', va='bottom', fontsize=8)
            ax.legend(**_KWARGS_LEGEND)

            if savefig: 
                fig.savefig('steadystate_sol.png', **_KWARGS_SAVEFIG)

            plt.show()

#==================================================================
#TRANSIENT SOLUTION
#==================================================================
def plotTranscendential(
          x_array: np.ndarray,
          f_array: np.ndarray,
          eigenvals: np.ndarray,
          fig_width: float,
          savefig: bool
          ) -> None:
    
    with plt.rc_context(_PLTCONTEXT):
        fig = plt.figure(figsize=(fig_width, fig_width/3))
        ax = fig.add_subplot(1, 1, 1)

        #plot lines-------------------------------------------
        ax.plot(x_array, f_array, color='#440154')
        ax.axhline(0.0, color='red', linestyle=':')
        for eigval in eigenvals: 
             ax.axvline(eigval, **_KWARGS_INTERFACES)

        #set limits------------------------------------------
        ax.set_ylim(f_array.min(), f_array.max())
        ax.set_xlim(eigenvals.min(), eigenvals.max())

        #add texts-------------------------------------------
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$f(\lambda)$')

        if savefig: 
             fig.savefig('transcendantial.png', **_KWARGS_SAVEFIG)

        plt.show()


def plotFullNorm(
          Nij: np.ndarray,
          print_results: bool,
          max_ticks: int,
          fig_width: float,
          savefig: bool
          ) -> None:
        
        #results printing----------------------------
        if print_results:
            min_diag = np.abs(Nij)[np.eye(Nij.shape[0], dtype=bool)].min()
            max_offdiag = np.abs(Nij)[~np.eye(Nij.shape[0], dtype=bool)].max()
            np.set_printoptions(formatter={'float': '{:.2e}'.format})
            print(f'Ratio min(diag)/max(off-diag): {min_diag/max_offdiag:.2e}')
            print(Nij)

        #data prepare-----------------------------------
        N_eigvals = Nij.shape[0]
        step = 1 if N_eigvals <=10 else max((5, N_eigvals//max_ticks))
        tickpos = np.arange(0, N_eigvals, step, dtype=float) + 0.5
        ticklabels = np.arange(0, N_eigvals, step, dtype=int)
        ticklabels[0] = 1
        ticklabels = ticklabels.astype(str)

        with plt.rc_context(_PLTCONTEXT):
            fig = plt.figure(figsize=(fig_width, fig_width))
            ax = fig.add_subplot(1, 1, 1)
            
            #plotting----------------------------------
            c = ax.pcolor(Nij, cmap='viridis')
            cbar = fig.colorbar(c, shrink=0.7)
            cbar.set_label(5*' ' + r'$N_{ij}$', rotation=0) #add a space in front

            #set limits--------------------------------
            ax.set_aspect('equal')
            
            #add texts---------------------------------
            ax.set_xlabel("i")
            ax.set_ylabel("j")
            ax.set_xticks(tickpos)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(tickpos)
            ax.set_yticklabels(ticklabels)

            if savefig: 
                 fig.savefig('orthogonality.png', **_KWARGS_SAVEFIG)

            plt.show()

def plotConvergence(
          Ni: np.ndarray,
          ImiPhim: np.ndarray,
          ImiGamm: np.ndarray,
          ImiFm: np.ndarray,
          max_ticks: int,
          fig_width: float,
          savefig: bool
          ) -> None:

        #data prepare---------------------------------------------
        N_eigvals = Ni.shape[0]
        x = np.arange(0, N_eigvals, 1)
        step = 1 if N_eigvals <=10 else max((5, N_eigvals//max_ticks))
        tickpos = np.arange(0, N_eigvals, step, dtype=float)
        ticklabels = np.arange(0, N_eigvals, step, dtype=int)
        ticklabels[0] = 1
        ticklabels = ticklabels.astype(str)
        cmap = _create_cmap(4)

        #plotting--------------------------------------------------
        with plt.rc_context(_PLTCONTEXT):
            fig = plt.figure(figsize=(fig_width, fig_width/3))
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(x, Ni, color = cmap[0], label=r'$N_i$')
            ax.plot(x, ImiPhim, color = cmap[1], label=r'$\sum_{m=1}^M \overline{I}_{m,i} \Phi_m$')
            ax.plot(x, ImiGamm, color = cmap[2], label=r'$\sum_{m=1}^M \overline{I}_{m,i} \Gamma_m$')
            ax.plot(x, ImiFm, color = cmap[3], label=r'$\sum_{m=1}^M \overline{I}_{m,i}\tilde{F}_m$')

            #set limits
            ax.set_xlim(x.min(), x.max())
            
            #add texts
            ax.set_xlabel("i")
            ax.set_xticks(tickpos)
            ax.set_xticklabels(ticklabels)
            ax.legend(**_KWARGS_LEGEND)

            if savefig: 
                 fig.savefig('convergence.png', **_KWARGS_SAVEFIG)

            plt.show()

#==============================================================
#CROSS PLOT THE PREDICTED AND MEASURED DATA
#==============================================================
def plotPredVSMeas(
          labels: np.ndarray,
          x_predicted: list[np.ndarray],
          T_predicted: list[np.ndarray],
          x_measured: list[np.ndarray],
          T_measured: list[np.ndarray],
          T_IC: np.ndarray,
          xm_1: np.ndarray,
          soil_types: np.ndarray| None = None,
          png_name: str|None = None,
          title: str|None = None,
          fig_width: float = 12.0
          ) -> None:
    
    #data prepare-------------------------------------------
    x_m = np.insert(xm_1,0,0) #all x_m, m=0,1 ... M
    cmap = _create_cmap(len(labels))

    with plt.rc_context(_PLTCONTEXT):
        fig = plt.figure(figsize=(fig_width, 6))
        ax = fig.add_subplot(1, 1, 1)

        #plotting---------------------------------------------
        for idx, label in enumerate(labels):
            #plot measrued
            ax.plot(T_measured[idx], x_measured[idx], color=cmap[idx], marker=_MARKERS[idx], label=label, **_KWARGS_MEAS)

            #plot predicted
            if idx == 0:
                ax.plot(T_IC, x_m, color = cmap[idx]) #plot the piecewise linear IC
            else:
                ax.plot(T_predicted[idx], x_predicted[idx], color = cmap[idx])

        #plot interfaces
        for i in xm_1: 
             ax.axhline(i, **_KWARGS_INTERFACES)

        #limits-----------------------------------------------
        ax.invert_yaxis()
        ax.set_ylim(x_m.max(), x_m.min())

        #texts------------------------------------------------
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel(r'GWT ($^{\circ}$C)')
        ax.legend(**_KWARGS_LEGEND)
        if title is not None: 
             ax.set_title(title)

        #add soil types
        if soil_types is not None:
            text_pos = (x_m[:-1] + x_m[1:])/2
            for idx, pos in enumerate(text_pos): 
                 ax.text(0.99, pos, soil_types[idx], color = 'gray', transform=ax.get_yaxis_transform(), horizontalalignment='right', verticalalignment='center')

        if png_name is not None:
            fig.savefig(f'{png_name}.png', **_KWARGS_SAVEFIG)
        
        plt.show()