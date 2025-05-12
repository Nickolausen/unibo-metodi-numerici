import sympy as sp
import numpy as np

def newton_raphson(x0, F, J, tolx, tolf, maxit):
    # {1} Select x_0
    # {2} Obtain f_0 = f(x_0) and J_0
    # {3} Obtain J_0^{-1}
    # {4} n = 0
    # {5} While || x_{n + 1} - x_n || > tol; do
    #     {5.1} x_{n+1} = x_n - J_n^{-1}·f_n
    #     {5.2} Find f_{n + 1} and J_{n + 1}
    #     {5.3} n = n + 1

    # {1} Select x_0
    x_curr = np.array(x0, dtype=float)
    attempts = [ ]
    erroreX = [ x_curr ]
    x_next = None
    stop = False

    while not stop:
        
        if np.linalg.det(J(*x_curr)) == 0:
            print("Jacobiana non invertibile!")
            return None, None, None

        # {5.1} x_{n+1} = x_n - J_n^{-1}·f_n
        x_next = (x_curr - (np.linalg.inv(J(*x_curr)) @ F(*x_curr)).T).flatten()
        attempts.append(x_next)

        error_curr = np.linalg.norm(x_next - x_curr, 1)
        error_curr /= np.linalg.norm(x_curr, 1) if np.linalg.norm(x_curr, 1) != 0 else 1
        erroreX.append(error_curr)

        # Criterio di arresto - x e y
        should_stop_x = error_curr <= tolx
        should_stop_f = np.all(F(*x_next) == 0) or np.all(np.abs(F(*x_next)) <= tolf)
        
        if should_stop_f or should_stop_x or len(attempts) > maxit:
            stop = True
        else:
            # Avanzo nella successione di iterati
            x_curr = x_next

    return attempts, len(attempts), erroreX

def newton_raphson_corde(x0, F, J, tolx, tolf, maxit):
    # {1} Select x_0
    # {2} Obtain f_0 = f(x_0) and J_0
    # {3} Obtain J_0^{-1}
    # {4} n = 0
    # {5} While || x_{n + 1} - x_n || > tol; do
    #     {5.1} x_{n+1} = x_n - J_n^{-1}·f_n
    #     {5.2} Find f_{n + 1} and J_{n + 1}
    #     {5.3} n = n + 1

    # {1} Select x_0
    x_curr = np.array(x0, dtype=float)
    attempts = [ ]
    erroreX = [ ]
    x_next = None
    stop = False
    J_const = J(*x_curr)
    
    if np.linalg.det(J_const) == 0:
        print("Jacobiana non invertibile!")
        return None, None, None
    
    while not stop:
        # {5.1} x_{n+1} = x_n - J_n^{-1}·f_n
        x_next = (x_curr - (np.linalg.inv(J_const) @ F(*x_curr)).T).flatten()
        attempts.append(x_next)

        error_curr = np.linalg.norm(x_next - x_curr, 1)
        error_curr /= np.linalg.norm(x_curr, 1) if np.linalg.norm(x_curr, 1) != 0 else 1
        erroreX.append(error_curr)

        # Criterio di arresto - x e y
        should_stop_x = error_curr <= tolx
        should_stop_f = np.all(F(*x_next) == 0) or np.all(np.abs(F(*x_next)) <= tolf)
        
        if should_stop_f or should_stop_x or len(attempts) > maxit:
            stop = True
        else:
            # Avanzo nella successione di iterati
            x_curr = x_next

    return attempts, len(attempts), erroreX