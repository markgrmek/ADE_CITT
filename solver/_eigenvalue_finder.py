import numpy as np
from scipy.optimize import brentq
from typing import Callable

def find_eigvals(
        trans_eq: Callable,
        num_eigvals: int,
        step: float = 0.5,
        tol: float = 1e-20
        ) -> np.ndarray:
    
    #FIND BRACKETS FOR FASTER CONVERGENCE
    brackets = []

    prev_x, prev_f = 0, trans_eq(0)
    while len(brackets) < num_eigvals:
        curr_x = prev_x + step
        curr_f = trans_eq(curr_x)

        cond1 = curr_f < 0 and 0 <= prev_f
        cond2 = prev_f < 0 and 0 <= curr_f

        if any((cond1, cond2)):
            brackets.append((prev_x, curr_x))

        prev_x, prev_f = curr_x, curr_f

    #FIND EIGENVALUES
    eigenvalues =[]
    for a,b in brackets:
        try:
            lamb = brentq(trans_eq, a, b, xtol=tol) 
            eigenvalues.append(lamb)
                    
        except (ValueError, RuntimeError):
            print(f"Failed to converge in bracket {a:.3e} to {b:.3e}")
            continue
    
    return np.array(eigenvalues)