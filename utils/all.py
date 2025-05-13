from scipy.io import loadmat
import numpy as np
import RisolviSis as RS
import scipy.linalg as spl
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
import math
import sympy as sym

# m = n
# piccola-densa
#     benCond (Gauss)
#     MalCond (QR)
#     Simm&defpos (Cholesky)
# grande-sparsa
#     diagdomi(jacobi,GS,GSSOR)
#     simm-defpos(GS,GSSOR,discGRAD,gradConj)
# m>n
# benCond-maxrank(eqnorm)
# mehCond-maxrank(QRLS)
# malCOnd-NOmaxrank(SVDLS)

dati = loadmat()
A = dati['A']
b = dati['b']

m, n = A.shape
print("dimensioni matrice: ", m, n)

nz = np.count_nonzero(A)/(n*m)
perc_nz = nz * 100
print("Percentuale elementi diversi da 0: ", perc_nz)

flag = A==A.T
if np.all(flag) == 0:
    print("Matrice non simmetrica")
else:
    print("matrice simmetrica")
    
#se simmetrica, per sapere se è definita positiva:
autovalori = np.linalg.eigvals(A)
def_pos = np.all(autovalori > 0)
print(def_pos)
#se torna False non è definita positiva

#cond = 400, abbastanza ben condizionata, uso fattorizzazione di Gauss
xesatta = np.ones_like(b)
PT,L,U=spl.lu(A)
P=PT.T
y,flag=RS.Lsolve(L,P@b)
if flag==0:
    xLU,flag1=RS.Usolve(U,y)
    
err_LU=np.linalg.norm(xLU-xesatta)/np.linalg.norm(xesatta)
print("Errore percentuale soluzione LU ",err_LU*100)

#se fosse stata mal condizionata, QR
xesatta = np.ones_like(b)
Q, R = spl.qr(A)
yy = Q.T@b
xqr, flag = RS.Usolve(R, yy)
err_QR = np.linalg.norm(xqr - xesatta) / np.linalg.norm(xesatta)
print("Errore percentuale soluzione con RQ", err_QR)


#diagonale dominante?
def verifica_dd(A):
    n = A.shape[0]
    flag = True
    for i in range(n):
        el_diag = np.abs(A[i, i])
        sum_extradiag = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        if el_diag < sum_extradiag:
            print("Matrice non a diagonale dominante")
            flag = False
            return flag
    return flag

dd = verifica_dd(A)
print("Matrice a diagonale dominante?: ", dd)

#dd, ma anche no

def gauss_seidel(A, b, x0, toll, it_max):
    errore = 1000
    d = np.diag(A)
    D = np.diag(d)
    E = np.tril(A, -1)
    F = np.triu(A, 1)
    M = D + E
    N = -F
    T = np.dot(np.linalg.inv(M), N)
    autovalori = np.linalg.eigvals(T)
    raggiospettrale = np.max(np.abs(autovalori))
    print("Raggio spettrale Gauss-Seidel: ", raggiospettrale)
    it = 0
    er_vet = []
    while it <= it_max and errore >= toll:
        temp = b - np.dot(F, x0)
        x, flag = RS.Lsolve(M, temp)
        errore = np.linalg.norm(x-x0) / np.linalg.norm(x)
        er_vet.append(errore)
        x0 = x.copy()
        it = it + 1
    return x, it, er_vet

def jacobi(A, b, x0, toll, it_max):
    errore = 1000
    d = np.diag(A)
    n = A.shape[0]
    invM = np.diag(1/d)
    E = np.tril(A, -1)
    F = np.triu(A, 1)
    N = -(E + F)
    T = np.dot(invM, N)
    autovalori = np.linalg.eigvals(T)
    raggiospettrale = np.max(np.abs(autovalori))
    print("Raggio spettrale jacobi", raggiospettrale)
    it = 0
    #xold = x0.copy()
    er_vet = []
    while it <= it_max and errore >= toll:
        x = (b + np.dot(N, x0)) / d.reshape(n, 1)
        errore = np.linalg.norm(x - x0) / np.linalg.norm(x)
        er_vet.append(errore)
        x0 = x.copy()
        it = it + 1
    return x, it, er_vet

def gauss_seidel_sor(A, b, x0, toll, it_max, omega):
    errore = 1000
    d = np.diag(A)
    D = np.diag(d)
    Dinv = np.diag(1/d)
    E = np.tril(A, -1)
    F = np.triu(A, 1)
    Momega = D + omega * E
    Nomega = (1 - omega) * D - omega * F
    T = np.dot(np.linalg.inv(Momega), Nomega)
    autovalori = np.linalg.eigvals(T)
    raggiospettrale = np.max(np.abs(autovalori))
    print("Raggio spettrale Gauss-Seidel SOR", raggiospettrale)
    M = D + E
    N = -F
    it = 0
    xold = x0.copy()
    xnew = x0.copy()
    er_vet = []
    while it <= it_max and errore >= toll:
        temp = b - np.dot(F, xold)
        xtilde, flag = RS.Lsolve(M, temp)
        xnew = (1 - omega) * xold + omega * xtilde
        errore = np.linalg.norm(xnew - xold) / np.linalg.norm(xnew)
        er_vet.append(errore)
        xold = xnew.copy()
        it = it + 1
    return xnew, it, er_vet

def conjugate_gradient(A,b,x0,itmax,tol):
#Metodo del gradiente coniugato per la soluzione di un sistema lineare con matrice dei coefficienti simmetrica e definita positiva
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]    
# inizializzare le variabili necessarie
    x = x0
    r = A.dot(x)-b
    p = -r
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x0)
    vet_r=[]
    vet_r.append(errore)
# utilizzare il metodo del gradiente coniugato per trovare la soluzione
    while errore >= tol and it< itmax:
        it=it+1
        Ap=A.dot(p)
        rtr=np.dot(r.T, r)
        alpha = rtr / np.dot(p.T, Ap)
        x = x + alpha *p
        vec_sol.append(x)
        r=r+alpha*Ap
        gamma=np.dot(r.T,r)/rtr
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p = -r+gamma*p
   
    return x,vet_r,vec_sol,it


def steepestdescent(A,b,x0,itmax,tol):
#Metodo del gradiente   per la soluzione di un sistema lineare con matrice dei coefficienti simmetrica e definita positiva
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]
    
   # inizializzare le variabili necessarie
    x = x0
    r = A.dot(x)-b
    p = -r
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x)
    vet_r=[]
    vet_r.append(errore)
     
    # utilizzare il metodo del gradiente per trovare la soluzione
    while errore>= tol and it< itmax:
        it=it+1
        Ap=A.dot(p)
        rTr=np.dot(r.T, r)
        alpha = rTr / np.dot(p.T, Ap)
        x = x + alpha*p
        vec_sol.append(x)
        r=r+alpha*Ap
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p = -r 
        
    return x,vet_r,vec_sol,it


x0 = np.zeros(A.shape[0]).reshape(n, 1)
toll = 1e-8
it_max = 100000
omega = 1.4 #valore per cui si ha il numero minimo di iterazioni.
solJac, itJac, err_vetJac = jacobi(A, b, x0, toll, it_max)
solGS, itGS, err_vetGS = gauss_seidel(A, b, x0, toll, it_max)
solGSor, itGSor, err_vetGSor = gauss_seidel_sor(A, b, x0, toll, it_max, omega)
print("Iterazioni Jac", itJac)
print("Iterazioni GS", itGS)
print("Iterazioni GSor", itGSor)








#se m != n

U, s, VT = spl.svd(A)
thresh = np. spacing(1) * m * s[0] #m=max(m,n)
k = np.count_nonzero(s > thresh)  #Calcolo del rango della matrice, numero dei valori singolari diversi maggiori della soglia
print("rango = ", k)
if k < n :
    print("La matrice non ha rango massimo")
else:
    print("La matrice ha rango massimo")
  
#se non ha rango massimo
def SVDLS(A, b):
    n = A.shape[1] #numero di colonne di A
    U, s, VT = spl.svd(A)
    V = VT.T
        
    thresh = np.spacing(1) * m * s[0]
    ##Calcolo del rango della matrice, numero dei valori singolari maggiori di una soglia
    k = np.count_nonzero(s > thresh)
    print("rango= ", k)
    if k < n:
        print("La matricce non è a rango massimo")
    else:
        print("La matrice è a rango massimo")
        d = U.T@b
        d1 = d[:k].reshape(k, 1)
        s1 = s[:k].reshape(k, 1)
        #Risolve il sistema diagonale di dimensione kxk avente come
        # matrice dei coefficienti la matrice Sigma
        c = d1 / s1
        x = V[:, :k]@c
        residuo = np.linalg.norm(d[k:])**2
        return x, residuo
                    

x, residuo = SVDLS(A, b)
print("Soluzione nel senso dei minimi quadrati ", x)
print("residuo: ", residuo)
print("Norma soluzione", np.linalg.norm(x))


#nel calcolo di una circonferenza
px = np.array([0, 4, 0, 5])
py = np.array([0, 0, 4, 6])

#Costruisco il sistema lineare sovradeterminato come richiesto
A = np.array([[0, 0 ,1], [4, 0, 1], [0, 4, 1], [5, 6, 1]])
b = np.array([[0], [-16], [-16], [-61]])

a, err = SVDLS(A, b)
print("Norma 2 al quadrato dell'errore: ", err)
#Calcolo il centro
cx = -a[0] / 2
cy = -a[1] / 2
#calcolo il raggio con la formula data
r1 = math.sqrt((a[0]**2) / 4 + (a[1]**2) / 4 - a[2])
t = np.linspace(0, 2 * math.pi, 100)
#costruisco le due componenti parametriche della circonferenza
x = cx + r1 * np.cos(t)
y = cy + r1 * np.sin(t)
plt.plot(x, y, 'r-')
plt.plot(px, py, 'go')
plt.axis('equal')



#oppure soluzione sistema sovradeterminato
def qrLS(A, b):
    n = A.shape[1]
    Q, R = spl.qr(A)
    h = Q.T@b
    x, flag = RS.Usolve(R[0:n, :], h[0:n])
    residuo = np.linalg.norm(h[n:])**2
    return x, residuo

#Soluzione di un sistema sovradeterminato facendo uso delle equazioni normali
def eqnorm(A, b):
    G = A.T@A
    print("Indice di condizionamento di G: ", np.linalg.cond(G))
    f = A.T@b
    L = spl.cholesky(G, lower = True)
    y, flag = RS.Lsolve(L, f)
    if flag == 0:
        x, fag = RS.Usolve(L.T, y)
        
    return x  

#coppie di dati sperimentali
x = dati["x"]
y = dati["y"]
m = x.shape[0]
x = x.reshape(m, )
y = y.reshape(m, )
plt.plot(x, y, 'ro')
n = 3
n1 = n + 1
A3 = np.vander(x, increasing = True)[:, :n1]
print("Rango: ", np.linalg.matrix_rank(A3), "Condizionamento: ", np.linalg.cond(A3))
alphaqr, res = qrLS(A3, y)
print("residuo: ", res)
xx = np.linspace(np.min(x), np.max(x), 200)
polQR = np.polyval(np.flip(alphaqr),xx)
plt.plot(xx, polQR)










#Funzioni per la costruzione del polinomio interpolatore nella base di
#Lagrange

def plagr(xnodi,k):
    """
    Restituisce i coefficienti del k-esimo pol di
    Lagrange associato ai punti del vettore xnodi
    """
    xzeri=np.zeros_like(xnodi)
    n=xnodi.size
    if k==0:
       xzeri=xnodi[1:n]
    else:
       xzeri=np.append(xnodi[0:k],xnodi[k+1:n])
    
    num=np.poly(xzeri) 
    den=np.polyval(num,xnodi[k])
    
    p=num/den
    
    return p

def Interpl(x, f, xx):
     """"
        %funzione che determina in un insieme di punti il valore del polinomio
        %interpolante ottenuto dalla formula di Lagrange.
        % DATI INPUT
        %  x  vettore con i nodi dell'interpolazione
        %  f  vettore con i valori dei nodi 
        %  xx vettore con i punti in cui si vuole calcolare il polinomio
        % DATI OUTPUT
        %  y vettore contenente i valori assunti dal polinomio interpolante
        %
     """
     n=x.size
     m=xx.size
     L=np.zeros((m,n))
     for k in range(n):
        p=plagr(x,k)
        L[:,k]=np.polyval(p,xx)
    
     return np.dot(L,f)
 
 
x = np.array([1, 1.5, 1.75])
f = lambda x: np.cos(np.pi*x) + np.sin(np.pi*x)
y = f(x)
xx = np.linspace(0, 2, 200)
polL = Interpl(x, y, xx)
plt.plot(xx, f(xx), 'r-', xx, polL, 'g--', x, y, 'bo')
plt.legend(['Esatta', 'polinomio interpolatore', 'Nodi di interpolazione'])

#costante di LEBESGUE
x = np.array([1, 1.5, 1.75])
xx = np.linspace(0, 2, 200)
n = 2 #grado polinomio interpolante
Ls = np.zeros((200, 1))
for j in range(n+1):
    pL = plagr(x, j)
    Ls = Ls + np.abs(np.polyval(pL, xx))
cL = np.max(Ls)
 
 
 
 
 
def Lsolve(L,b):
#test dimensione
    m,n=L.shape
    flag=0;
    if n != m:
        print('errore: matrice non quadrata')
        flag=1
        x=[]
        return x, flag
    
# Test singolarita'
    if np.all(np.diag(L)) != True:
         print('el. diag. nullo - matrice triangolare inferiore')
         x=[]
         flag=1
         return x, flag
# Preallocazione vettore soluzione
    x=np.zeros((n,1))
    
    for i in range(n):
         s=np.dot(L[i,:i],x[:i]) #scalare=vettore riga * vettore colonna
         x[i]=(b[i]-s)/L[i,i]
      
    return x,flag

def Usolve(U,b):
#test dimensione
    m,n=U.shape
    flag=0;
    if n != m:
        print('errore: matrice non quadrata')
        flag=1
        x=[]
        return x, flag
    
     # Test singolarita'
    if np.all(np.diag(U)) != True:
         print('el. diag. nullo - matrice triangolare superiore')
         x=[]
         flag=1
         return x, flag
    # Preallocazione vettore soluzione
    x=np.zeros((n,1))
    
    for i in range(n-1,-1,-1):
         s=np.dot(U[i,i+1:n],x[i+1:n]) #scalare=vettore riga * vettore colonna
         x[i]=(b[i]-s)/U[i,i]
      
    return x,flag


 
 
 
 
#lambda 
 
x = 10.0**np.arange(0, 21)
f = lambda x : 1/x - 1/(x+0.04)
fx = f(x)
s = sym.symbols('s')
fs = 1/s - 1/(s+0.04)
dfs = sym.diff(fs, s, 1) #CALCOLO DELLA DERIVATA
df_numerica = lambdify(s, dfs, np)
f_numerica = lambdify(s, fs, np)
indice_condizionamento = np.abs(df_numerica(x)*x/f_numerica(x))
spacing = np.spacing(x)
 
 
 
 
 
 
 
 
 
 
 
 #esercizio con retta di regressione
 # Per i dati (xi, yi) riportati nei seguenti array
# x = np.array([0.0004, 0.2507, 0.5008, 2.0007, 8.0013]) 
# y = np.array([0.0007,0.0162, 0.0288, 0.0309, 0.0310])

# - costruire la retta di regressione;
# - costruire la parabola approssimante i dati nel senso dei minimi quadrati;
# - determinare l'approssimazione ai minimi quadrati espressa in termini di
# basi esponenziali: y = a + be^{-x}+ ce^{-2x}
m = 5
x4 = np.array([0.0004, 0.2507, 0.5008, 2.0007, 8.0013])
y4 = np.array([0.0007, 0.0162, 0.0288, 0.0309, 0.0310])
M = np.zeros((5, 3))
M[:, 0] = np.ones((5, ))
M[:, 1] = np.exp(-x4)
M[:, 2] = np.exp(-2 * x4)

print("Rango: ", np.linalg.matrix_rank(M))
print("Condizionamento M: ", np.linalg.cond(M))

#La matrice M è a rango massimo, ha condizionamento 18.45, quindi
# la marice G delle equazioni
#normali (di dimensioni 4x4) avrebbe indice
#di condizionamento circa 343. Utilizzo il metodo

aexp, resexp = qrLS(M, y4)

xx = np.linspace(np.min(x4), np.max(x4), 200)
polexp = aexp[0] + aexp[1] * np.exp(-xx) + aexp[2] * np.exp(-2 * xx)
#calcolo del polinomio approssimante di grado 1
n = 1
n1 = n + 1
A1 = np.vander(x4, increasing = True)[:, :n1]
print("Rango di A1: ", np.linalg.matrix_rank(A1))
print("Condizionamento di A1: ", np.linalg.cond(A1))
#matrice di rango massimo e ben condizionata:
#uso le equazioni normali
alpha1 = eqnorm(A1, y4)
pol1 = np.polyval(np.flip(alpha1), xx)
 
#parabola approssimante
n = 2
n1 = n + 1
A2 = np.vander(x4, increasing = True)[:, :n1]
print("Rando A2: ", np.linalg.matrix_rank(A2))
print("condizionamento A2: ", np.linalg.cond(A2))
#matrice a rango massimo e mediamente mal condizionata:
#uso qr
alpha2, res2 = qrLS(A2, y4)
pol2 = np.polyval(np.flip(alpha2), xx)
print("quadrato residuo exp", resexp)
print("quadrato residuo pol grado 2: ", res2)

plt.plot(x4, y4, 'ro', xx, polexp, 'b--', xx, pol1, 'g:', xx, pol2, 'm')
plt.legend(['Osservazioni','Esponenziale','Polinomio grado 1','Polinomio grado 2'])