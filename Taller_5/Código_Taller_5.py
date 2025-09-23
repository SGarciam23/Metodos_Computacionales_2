# ------------------------------
# PUNTO 1
# ------------------------------

# ------------------------------
# 1.a)
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

