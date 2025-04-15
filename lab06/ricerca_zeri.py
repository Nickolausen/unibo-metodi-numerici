def bisezione(a, b, fname, tolx, tolf, nmax):
    """
    Calcola lo zero di una funzione via bisezione.
    """

    if (fname(a) * fname(b) > 0):
        print("Ipotesi del teorema dell'esistenza degli zeri non rispettata!")
        return (None, None, None)
    
    a_k = a
    b_k = b
    c = (a_k + b_k) / 2
    attempts = [c]
    
    def should_stop():
        return len(attempts) > nmax or abs(fname(c)) <= tolf or abs(b_k - a_k) <= tolx
    
    while (not should_stop()):
        res = fname(c)
        if (abs(fname(c)) <= tolf): 
            break
        if (res < 0):
            a_k = c
        else:
            b_k = c
        c = (a_k + b_k) / 2
        attempts.append(c)

    return (c, len(attempts), attempts)

def falsi(a, b, fname, tolx, tolf, nmax):
    """
    Calcola lo zero di una funzione via regula falsi.
    """

    if (fname(a) * fname(b) >= 0):
        print("Ipotesi del teorema dell'esistenza degli zeri non rispettata!")
        return (None, None, None)
    
    a_k = a
    b_k = b
    x_k = a_k - fname(a_k) * ((b_k - a_k) / (fname(b_k) - fname(a_k)))
    attempts = [x_k]

    def should_stop():
        # to change — abs(b_k - a_k) <= tolx
        return len(attempts) > nmax or abs(fname(x_k)) <= tolf or abs(b_k - a_k) <= tolx
    
    while (not should_stop()):
        res = fname(x_k)
        if (abs(fname(x_k)) <= tolf):
            break
        if (res < 0):
            a_k = x_k
        else:
            b_k = x_k
        x_k = a_k - fname(a_k) * ((b_k - a_k) / (fname(b_k) - fname(a_k)))
        attempts.append(x_k)

    return (x_k, len(attempts), attempts)
