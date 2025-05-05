import sympy as sp
import numpy as np

def sign(x):
    if x == 0:
        return 0
    return abs(x) / x

def bisezione(f, a, b, tolx, tolf, maxit):
    """
    Convergenza globale
    """

    # Se non viene rispettata l'ipotesi del Teorema dell'esistenza degli zeri, allora termina subito!
    if (sign(f(a)) * sign(f(b)) >= 0):
        return None, None, None
    
    a_k = a
    b_k = b
    guess = None
    
    attempts = [ ]
    stop = False
    
    while not stop:
        # Calcolo un nuovo punto medio sull'intervallo [a_k, b_k]
        guess = (a_k + b_k) / 2
        
        # Segnalo che il punto medio è un tentativo che potrebbe rappresentare la soluzione cercata 
        attempts.append(guess)
        
        # Valuto la funzione sul nuovo punto medio
        result = f(guess)

        # Stop conditions
        if result == 0 or abs(result) <= tolf or abs(a_k - b_k) <= tolx or len(attempts) > maxit:
            stop = True
        else:
            # Aggiornamento dell'intervallo per la prossima iterazione
            if sign(result) < 0:
                a_k = guess
            else:
                b_k = guess

    return guess, len(attempts), attempts

def regula_falsi(f, a, b, tolx, tolf, maxit):
    """
    Convergenza globale
    """
    
    # Se non viene rispettata l'ipotesi del Teorema dell'esistenza degli zeri, allora termina subito!
    if (sign(f(a)) * sign(f(b)) >= 0):
        return None, None, None
    
    # Estremi dell'intervallo in cui viene ricercato lo zero
    a_k = a
    b_k = b
    
    # x interna all'intervallo [a_k, b_k] che, ad ogni iterazione,
    # viene valutata dalla funzione per determinare se rappresenta lo zero cercato
    guess = None
    attempts = [ ]
    stop = False
    
    while not stop:
        # Calcolo la x: a differenza della bisezione, qui si considera la 
        # x ottenuta dall'intersezione della retta che congiunge i due estremi
        # [a_k, b_k] con l'asse x.

        # Calcolo il coefficiente angolare della retta
        m_k = (f(b_k) - f(a_k)) / (b_k - a_k)
        
        # Stessa formula di 'corde', 'secanti' e 'newton'
        guess = a_k - f(a_k) / m_k
        attempts.append(guess)
        result = f(guess)

        # Stop conditions
        if result == 0 or abs(result) <= tolf or abs(a_k - b_k) <= tolx or len(attempts) >= maxit:
            stop = True
        else:
            # Aggiornamento dell'intervallo per la prossima iterazione
            if sign(result) < 0:
                a_k = guess
            else:
                b_k = guess

    return guess, len(attempts), attempts

def corde(f, a, b, x0, tolx, tolf, maxit):
    """
    Convergenza locale - necessario parametro x0 per poter iniziare
    """
    m = (f(b) - f(a))/(b - a)
    x_curr = x0
    x_next = None
    attempts = [  ]
    stop = False
    
    while not stop:
        x_next = x_curr - f(x_curr) / m
        attempts.append(x_next)
        
        result = f(x_next)
        if result == 0 or len(attempts) >= maxit:
            stop = True
        else:
            # Progredisco nella successione di x_k
            x_curr = x_next

    return x_next, len(attempts), attempts

def secanti(f, x0, x1, tolx, tolf, maxit):
    """
    Convergenza locale - necessari parametri x0 e x1 per poter iniziare
    """
    # Come per gli altri algoritmi, iterazione dopo iterazione si genererà una sequenza di x
    # su cui viene valutata la funzione per poter determinare lo zero ricercato.
    # In questo caso, di questa successione serve memorizzare, per un passo k, anche il passo k-1 e k+1.
    # Abbiamo quindi:
    # · x_prev = x_{k-1}
    # · x_curr = x_k
    # · x_next = x_{k+1}
    
    x_prev, x_curr = x0, x1
    x_next = None
    attempts = [ ]
    stop = False
    
    while not stop:
        m_k = (f(x_curr) - f(x_prev)) / (x_curr - x_prev)
        x_next = x_curr - f(x_curr) / m_k
        attempts.append(x_next)
        
        result = f(x_next)
        if result == 0 or len(attempts) >= maxit or abs(x_curr - x_next) <= tolx or abs(result) <= tolf:
            stop = True
        else:
            # Progredisco nella successione di x_k
            x_prev, x_curr = x_curr, x_next

    return x_next, len(attempts), attempts

def newton(f_expr, x0, tolx, tolf, maxit):
    """
    Convergenza locale - necessario parametro x0 per poter iniziare 
    """
    x = sp.Symbol("x")
    
    df_expr = sp.diff(f_expr, x)
    
    f = sp.lambdify(x, f_expr, 'numpy')
    df = sp.lambdify(x, df_expr, 'numpy')

    x_curr = x0
    x_next = None
    attempts = [ ]
    stop = False

    while not stop:
        # Controllo divisione per zero (o intorno)
        if df(x_curr) <= np.spacing(1):
            return None, None, None
        
        x_next = x_curr - f(x_curr) / df(x_curr)
        attempts.append(x_next)
        
        result = f(x_next)
        if result == 0 or len(attempts) >= maxit or abs(x_curr - x_next) <= tolx or abs(result) <= tolf:
            stop = True
        else:
            x_curr = x_next
        
    return x_next, len(attempts), attempts