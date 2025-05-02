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