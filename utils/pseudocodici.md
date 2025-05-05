## Soluzione equazioni non lineari

### Bisezione

### Regula Falsi
1) Se $f(a) \cdot f(b) < 0$, allora: $a_0 = a,\quad b_0 = b$
2) while (!arresto):
3) 
$$
x_{k + 1} = a_k - f(a_k) \cdot \frac{b_k - a_k}{f(b_k) - f(a_k)}
$$
4) Se $f(x_{k + 1}) \cdot f(a_k) < 0$, allora $a_{k + 1} = a_k, \quad b_{k + 1} = x_{x + 1}$
5) Altrimenti, se $f(x_{k + 1}) \cdot f(b_k) < 0$, allora $a_{k + 1} = x_{x + 1}, \quad b_{k + 1} = b_k$
6) se $f(x_{k + 1}) = 0$, finito!
### Linearizzazione

#### $m_k$ costante — metodo delle corde

· Scelgo $m_k = \frac{f(b) - f(a)}{b - a}$

#### $m_k$ non costante — metodo delle secanti

· Scelgo $m_k = \frac{f(x_k) - f(x_{k - 1})}{x_k - x_{k - 1}}$

#### $m_k$ non costante — metodo di Newton

· Scelgo $m_k = f'(x)$

## Soluzione di sistemi lineari

### Metodi diretti

#### Forward substitution
Partendo da 
- sistema `Lx = b`, dove *L* è una **matrice triangolare inferiore** (*L = lower*)

```pseudo
for i = 1,2,...,n
    x_i = b_i
    for j = 1,2,…,i-1
        x_i = x_i - l_ij * x_j
    end for j
    x_i = x_i / l_ii
end for i
```

> Complessità computazionale: $O(\frac{n^2}{2})$

#### Backward substitution

Partendo da
- sistema `Ux = b`, dove *U* è una **matrice triangolare superiore** (*U = upper*)

```pseudo
for i=n,n-1,...,1
    x_i = b_i
    for j=i+1,…,n
        x_i = x_i - u_ij * x_j
    end for j
    x_i = x_i / u_ii
end for i
```

> Complessità computazionale: $O(\frac{n^2}{2})$