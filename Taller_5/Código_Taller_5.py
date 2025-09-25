# ------------------------------
# PUNTO 1
# ------------------------------

# ------------------------------
# 1.a) UN SOLO BETHA
# ------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


N = 500          
J = 1.0         
beta = 0.5        
steps = 2000000    

np.random.seed(42)
spins = np.random.choice([-1, 1], size=(N, N))


def energia_total(spins):
    E = 0
    for i in range(N):
        for j in range(N):
            s = spins[i, j]
            E -= J * s * (spins[(i+1)%N, j] + spins[(i-1)%N, j] +
                          spins[i, (j+1)%N] + spins[i, (j-1)%N])
    return E / 2  # cada par contado dos veces

#Intento del bono
def mean_cluster_area(spins, target_spin=1, exclude_largest=False):
    mask = (spins == target_spin)
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])  # conectividad 4
    labeled, n_clusters = ndimage.label(mask, structure=structure)
    if n_clusters == 0:
        return 0.0
    counts = np.bincount(labeled.ravel())[1:]  # quitar fondo
    if exclude_largest and counts.size > 1:
        counts = counts[counts != counts.max()]
    return counts.mean() if counts.size > 0 else 0.0


E = energia_total(spins)
M = spins.sum()

energias = []
magnetizaciones = []
clusters_mean = []

check_interval = (N*N) // 10


for step in range(steps):
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)

    s = spins[i, j]
    vecinos = (spins[(i+1)%N, j] + spins[(i-1)%N, j] +
               spins[i, (j+1)%N] + spins[i, (j-1)%N])
    dE = 2 * J * s * vecinos

    if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
        spins[i, j] *= -1
        E += dE
        M += 2 * spins[i, j]

    if step % check_interval == 0:
        energias.append(E / (4 * N * N))
        magnetizaciones.append(M / (N * N))
        
        clusters_mean.append(mean_cluster_area(spins, target_spin=1))


plt.figure(figsize=(8,5))
plt.plot(energias, label="Energía normalizada")
plt.plot(magnetizaciones, label="Magnetización normalizada")
plt.xlabel("Iteraciones (x10⁴)")
plt.ylabel("Valor normalizado")
plt.legend()
plt.title("Evolución del modelo de Ising 2D")
plt.tight_layout()
plt.savefig("1a_resultados.pdf")
plt.close()

# -----------------------------
# Gráfica tamaño promedio de clusters
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(clusters_mean, label="Tamaño promedio de clusters (+1)")
plt.xlabel("Iteraciones (x10⁴)")
plt.ylabel("Área promedio")
plt.legend()
plt.title("Evolución del tamaño promedio de clusters")
plt.tight_layout()
plt.savefig("clusters_mean.pdf")
plt.close()

# ------------------------------
# 1.a) VARIOS BETHA
# ------------------------------

import numpy as np 
import matplotlib.pyplot as plt

# ------------------------------
# Parámetros del sistema
# ------------------------------
N = 100              # tamaño de la red (más pequeño para ver dispersión en M)
J = 1.0              # constante de acoplamiento
steps = 500000       # número de pasos Monte Carlo
beta = 0.5           # temperatura fija
n_trayectorias = 20  # cuántas simulaciones correr

# ------------------------------
# Función energía total
# ------------------------------
def energia_total(spins):
    E = 0
    for i in range(N):
        for j in range(N):
            s = spins[i, j]
            E -= J * s * (spins[(i+1)%N, j] + spins[(i-1)%N, j] +
                          spins[i, (j+1)%N] + spins[i, (j-1)%N])
    return E / 2  # cada enlace contado dos veces

# ------------------------------
# Simulación para varias trayectorias
# ------------------------------
plt.figure(figsize=(9,5))

for t in range(n_trayectorias):
    spins = np.random.choice([-1, 1], size=(N, N))
    E = energia_total(spins)
    M = spins.sum()

    energias = [0.0]          # arranque en (0,0)
    magnetizaciones = [0.0]
    epocas = [0]

    for step in range(steps):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        
        s = spins[i, j]
        vecinos = (spins[(i+1)%N, j] + spins[(i-1)%N, j] +
                   spins[i, (j+1)%N] + spins[i, (j-1)%N])
        
        dE = 2 * J * s * vecinos
        
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1
            E += dE
            M += 2 * spins[i, j]
        
        # muestreo cada ~N*N/10 pasos
        if step % (N*N // 10) == 0:
            energias.append(E / (4 * N * N))          # normalización por sitio
            magnetizaciones.append(M / (N * N))
            epocas.append(step)

    # graficar trayectorias
    plt.plot(epocas, energias, color="black", alpha=0.5)
    plt.plot(epocas, magnetizaciones, color="red", alpha=0.5)

# ------------------------------
# Gráfica final
# ------------------------------
plt.xlabel("Épocas")
plt.ylabel("")
plt.title("Durante")
plt.xticks(np.arange(0, steps+1, 50000))  # ticks cada 50k
plt.xlim(0, steps)  # forzar que empiece exactamente en x=0
plt.tight_layout()
plt.savefig("1.a.pdf")



import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import math

# ------------------------------
# Parámetros del sistema
# ------------------------------
N = 100            # tamaño de la red
J = 1
sweeps_eq = 500    # sweeps para equilibrar
sweeps_meas = 1000 # sweeps para medir
betas = np.linspace(0.1, 0.9, 80)  # rango de betas

# Inicialización
np.random.seed(42)
espines = np.random.choice([-1, 1], size=(N, N))

# ------------------------------
# Funciones del modelo
# ------------------------------
@njit
def energia_total(spins, N, J):
    E = 0.0
    for i in range(N):
        for j in range(N):
            s = spins[i, j]
            E -= J * s * (spins[(i+1)%N, j] + spins[(i-1)%N, j] +
                          spins[i, (j+1)%N] + spins[i, (j-1)%N])
    return E/2.0   # cada enlace contado dos veces

@njit
def sweep(spins, N, J, beta, E, M):
    for _ in range(N*N):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s = spins[i, j]
        vecinos = spins[(i+1)%N, j] + spins[(i-1)%N, j] + spins[i, (j+1)%N] + spins[i, (j-1)%N]
        dE = 2 * J * s * vecinos
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] = -s
            E += dE
            M += 2 * spins[i, j]
    return E, M

@njit
def run_metropolis(spins, N, J, beta, sweeps_eq, sweeps_meas):
    E = energia_total(spins, N, J)
    M = spins.sum()
    
    # equilibrar
    for _ in range(sweeps_eq):
        E, M = sweep(spins, N, J, beta, E, M)
    
    # medir
    E_vals = np.empty(sweeps_meas)
    for k in range(sweeps_meas):
        E, M = sweep(spins, N, J, beta, E, M)
        E_vals[k] = E / (N*N)   # energía por espín
    
    return spins, E_vals

# ------------------------------
# Simulación sobre varios betas
# ------------------------------
Cv_vals = []
for beta in betas:
    espines, E_vals = run_metropolis(espines, N, J, beta, sweeps_eq, sweeps_meas)
    meanE = E_vals.mean()
    meanE2 = (E_vals**2).mean()
    # fórmula correcta con energía por espín
    Cv = beta*2 * (meanE2 - meanE*2)
    Cv_vals.append(Cv)

# ------------------------------
# Graficar
# ------------------------------
beta_c = 0.5 * math.log(1 + math.sqrt(2))

plt.figure(figsize=(8,5))
plt.plot(betas, Cv_vals, '-k')
plt.axvline(beta_c, color='red', linestyle='--', label="Critical point (theory)")
plt.xlabel("Thermodynamic β")
plt.ylabel("Specific heat from simulation")
plt.title(f"Specific heat vs β (N={N})")
plt.legend()
plt.tight_layout()
plt.savefig("1.b.pdf")
plt.show()

# ------------------------------
# 1.b)
# ------------------------------

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import math

# ------------------------------
# Parámetros del sistema
# ------------------------------
N = 100            # tamaño de la red
J = 1
sweeps_eq = 500    # sweeps para equilibrar
sweeps_meas = 1000 # sweeps para medir
betas = np.linspace(0.1, 0.9, 80)  # rango de betas

# Inicialización
np.random.seed(42)
espines = np.random.choice([-1, 1], size=(N, N))

# ------------------------------
# Funciones del modelo
# ------------------------------
@njit
def energia_total(spins, N, J):
    E = 0.0
    for i in range(N):
        for j in range(N):
            s = spins[i, j]
            E -= J * s * (spins[(i+1)%N, j] + spins[(i-1)%N, j] +
                          spins[i, (j+1)%N] + spins[i, (j-1)%N])
    return E/2.0   # cada enlace contado dos veces

@njit
def sweep(spins, N, J, beta, E, M):
    for _ in range(N*N):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s = spins[i, j]
        vecinos = spins[(i+1)%N, j] + spins[(i-1)%N, j] + spins[i, (j+1)%N] + spins[i, (j-1)%N]
        dE = 2 * J * s * vecinos
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] = -s
            E += dE
            M += 2 * spins[i, j]
    return E, M

@njit
def run_metropolis(spins, N, J, beta, sweeps_eq, sweeps_meas):
    E = energia_total(spins, N, J)
    M = spins.sum()
    
    # equilibrar
    for _ in range(sweeps_eq):
        E, M = sweep(spins, N, J, beta, E, M)
    
    # medir
    E_vals = np.empty(sweeps_meas)
    for k in range(sweeps_meas):
        E, M = sweep(spins, N, J, beta, E, M)
        E_vals[k] = E / (N*N)   # energía por espín
    
    return spins, E_vals

# ------------------------------
# Simulación sobre varios betas
# ------------------------------
Cv_vals = []
for beta in betas:
    espines, E_vals = run_metropolis(espines, N, J, beta, sweeps_eq, sweeps_meas)
    meanE = E_vals.mean()
    meanE2 = (E_vals**2).mean()
    # fórmula correcta con energía por espín
    Cv = beta*2 * (meanE2 - meanE*2)
    Cv_vals.append(Cv)

# ------------------------------
# Graficar
# ------------------------------
beta_c = 0.5 * math.log(1 + math.sqrt(2))

plt.figure(figsize=(8,5))
plt.plot(betas, Cv_vals, '-k')
plt.axvline(beta_c, color='red', linestyle='--', label="Critical point (theory)")
plt.xlabel("Thermodynamic β")
plt.ylabel("Specific heat from simulation")
plt.title(f"Specific heat vs β (N={N})")
plt.legend()
plt.tight_layout()
plt.savefig("1.b.pdf")
plt.show()
