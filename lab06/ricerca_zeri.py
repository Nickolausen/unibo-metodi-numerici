def sign(x):
    """
    Funzione segno: restituisce
    · +1    se x > 0
    · -1    se x < 0
    · 0     se x = 0
    """
    if x == 0:
        return 0
    return abs(x) / x

def bisezione(a, b, fname, tolx, tolf, nmax):
    """
    Calcola lo zero di una funzione via bisezione.
    """

    if (sign(fname(a)) * sign(fname(b)) > 0):
        print("Ipotesi del teorema dell'esistenza degli zeri non rispettata!")
        return None, None, None
    
    a_k = a
    b_k = b
    c = (a_k + b_k) / 2
    attempts = [ c ]
    
    def should_stop():
        return len(attempts) > nmax or\
            abs(fname(c)) <= tolf or\
            abs(b_k - a_k) <= tolx
    
    while not should_stop():
        res = fname(c)
        if (abs(fname(c)) <= tolf): 
            break
        if (res < 0):
            a_k = c
        else:
            b_k = c
        c = (a_k + b_k) / 2
        attempts.append(c)

    return c, len(attempts), attempts

def falsi(a: float, b: float, fname, tolx: float, tolf: float, nmax: int):
    """
    Calcola lo zero di una funzione via regula falsi.
    """

    if sign(fname(a)) * sign(fname(b)) >= 0:
        print("Ipotesi del teorema dell'esistenza degli zeri non rispettata!")
        return None, None, None
    
    a_k = a
    b_k = b
    x_k = a_k - fname(a_k) * (b_k - a_k) / (fname(b_k) - fname(a_k))
    attempts = [ x_k ]
    x_prec = x_k + 1

    def should_stop():
        error = abs(x_k - x_prec) / abs(x_k) if x_k != 0 else abs(x_k - x_prec)
        return len(attempts) > nmax or\
            abs(fname(x_k)) <= tolf or\
            error <= tolx

    while not should_stop():
        res = fname(x_k)
        print(abs(fname(x_k)) <= tolf)
        if res == 0 or abs(fname(x_k)) <= tolf:
            break
        if sign(res) * sign(fname(a_k)) < 0:
            b_k = x_k
        elif sign(res) * sign(fname(b_k)) < 0:
            a_k = x_k
        x_prec = x_k
        x_k = a_k - fname(a_k) * ((b_k - a_k) / (fname(b_k) - fname(a_k)))
        attempts.append(x_k)

    return x_k, len(attempts), attempts
