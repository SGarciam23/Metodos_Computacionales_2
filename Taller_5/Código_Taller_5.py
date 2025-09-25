# ------------------------------
# PUNTO 1
# ------------------------------

# ------------------------------
# 1.a) UN SOLO BETHA
# ------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.integrate import solve_ivp


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
    Cv = beta**2 * (meanE2 - meanE**2)
    Cv_vals.append(Cv)


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
# PUNTO 2
# ------------------------------
# Importo numpy y matplotlib que faltaban


#-------------------------------
# Parte A: solución determinista (corregida)
#-------------------------------
# Defino los parámetros fijos del problema: tasa de creación A y tasa de extracción B
A = 1000.0  # tasa de creación de 239U por día
B = 20.0    # tasa de extracción del Pu por día

# Defino las vidas medias dadas en el enunciado y convierto a días
t12_U_minutes = 23.4                     # vida media de 239U en minutos
t12_U_days = t12_U_minutes / 1440.0      # convierto minutos -> días
t12_Np_days = 2.36                       # vida media de 239Np en días

# Calculo las constantes de decaimiento lambda = ln(2) / t1/2 en unidades de 1/día
lambda_U = np.log(2) / t12_U_days
lambda_Np = np.log(2) / t12_Np_days

# Defino el sistema de ecuaciones diferenciales (derivadas)
def dydt(t, y):
    # y = [U, Np, Pu]
    U, Np, Pu = y
    dU = A - lambda_U * U
    dNp = lambda_U * U - lambda_Np * Np
    dPu = lambda_Np * Np - B * Pu
    return [dU, dNp, dPu]

# Defino la tolerancia para considerar que el sistema llegó a estado estable
tol_derivative = 1e-3  # umbral en unidades por día para la magnitud de las derivadas

# Defino un evento que detecta cuando la magnitud máxima de las derivadas cae por debajo de tol_derivative
def event_steady(t, y):
    derivs = np.abs(dydt(t, y))
    return np.max(derivs) - tol_derivative

# Indico que el evento es terminal (detiene la integración) y que detectamos cualquier cruce descendente
event_steady.terminal = True
event_steady.direction = -1

# Condiciones iniciales: 10 unidades de cada isotopo al inicio
y0 = [10.0, 10.0, 10.0]  # [U(0), Np(0), Pu(0)]

# Intervalo de tiempo a simular: 30 días
t_span = (0.0, 30.0)
t_eval = np.linspace(t_span[0], t_span[1], 3000)

# Ejecuto la integración usando solve_ivp con el evento de estado estable
sol = solve_ivp(dydt, t_span, y0, t_eval=t_eval, events=event_steady, dense_output=True, atol=1e-8, rtol=1e-6)

# Compruebo si el evento de estado estable ocurrió y determino el tiempo (si ocurrió)
if sol.t_events and len(sol.t_events[0]) > 0:
    t_steady = float(sol.t_events[0][0])
    print(f"Estado estable detectado en t = {t_steady:.6f} días (umbral derivada = {tol_derivative}).")
else:
    t_steady = None
    print("No se detectó estado estable en los 30 días simulados con la tolerancia dada.")

# Extraigo la solución para cada especie (U, Np, Pu)
U = sol.y[0]
Np = sol.y[1]
Pu = sol.y[2]
t = sol.t

# Cálculo analítico del estado estacionario para verificar
U_ss_anal = A / lambda_U
Np_ss_anal = A / lambda_Np
Pu_ss_anal = A / B
print(f"Estado estacionario analítico aproximado: U={U_ss_anal:.3f}, Np={Np_ss_anal:.3f}, Pu={Pu_ss_anal:.3f}")

# Grafico las tres cantidades en función del tiempo (escala lineal para ver comportamiento inicial)
plt.figure(figsize=(8, 5))
plt.plot(t, U, label='239U (U)', color='C0')
plt.plot(t, Np, label='239Np (Np)', color='C1')
plt.plot(t, Pu, label='239Pu (Pu)', color='C2')
plt.xlabel('Tiempo (días)')
plt.ylabel('Cantidad (unidades)')
plt.xscale('log') # escala logarítmica en x para ver mejor el inicio
plt.yscale('log') # escala logarítmica en y para ver mejor el inicio
plt.title('Evolución determinista de U, Np y Pu (30 días)')
plt.legend()
plt.grid(True)

# Si se detectó estado estable, marco el tiempo en la gráfica
if t_steady is not None:
    plt.axvline(t_steady, color='k', linestyle='--', label=f'Est. estable t={t_steady:.3f} d')
    plt.legend()

plt.tight_layout()
plt.savefig('Taller_5/2.a.pdf', dpi=300)
#plt.show()


# -------------------------------
# Parte B: Ecuación diferencial estocástica SOLO para U(t) (RK2 estocástico)
# -------------------------------

#PUNTO 2.B
def RK2_U (A, lambdau, U_0, T, dt, n_traj):
    #numero de pasos
    n_steps = int(T/dt)
    #arreglo para la cantidad de Uranio
    U = np.zeros((n_traj, n_steps))
    #Arrays de tiempos
    t = np.linspace(0, T, n_steps+1)
    #Array de trayectorias en cero
    traj = np.zeros((n_traj, n_steps+1))
    #valor inicial
    U[:, 0] = U_0
    #Función mu y sigma_mu
    mu = lambda U: A - lambdau * U 
    sigmamu = lambda U: np.sqrt(np.maximum(A + lambdau*U,0.0))
    #simulación de las trayectorias
    #Iniciar una nueva trayectoria
    for j in  range(n_traj):
        U = U_0
        #simular trajectoria
        for n in range(n_steps):
            #valores de ruido
            W = np.random.normal(0.0, 1.0)
            S = np.random.choice([-1, 1])
            Noise = (W-S) * np.sqrt(dt)
            NNoise = (W+S) * np.sqrt(dt)
            #K1
            K_1 = dt * mu(U) + sigmamu(U) * Noise
            #K2
            K_2 = dt * mu(U + K_1) + NNoise * sigmamu(U + K_1)
            #Actualizar U para el siguiente paso
            U = U + 0.5 * (K_1 + K_2)

            #Guardar el valor de U
            traj[j, n+1] = U
    return t, traj
    

#simular
#prueba 1
A=1000
lambdaU = np.log(2)/0.01625
U_0 = 10
#numero de dias
T = 30
#Longitud del paso
dt = 0.0001
#trayectorias a simular
n_traj = 5
tiempos, trayectorias = RK2_U(A, lambdaU, U_0, T, dt, n_traj)

plt.plot(tiempos, trayectorias[0], label='U ruta 1')
plt.plot(tiempos, trayectorias[1], label='U ruta 2')
plt.plot(tiempos, trayectorias[2], label='U ruta 3')
plt.plot(tiempos, trayectorias[3], label='U ruta 4')
plt.plot(tiempos, trayectorias[4], label='U ruta 5')
plt.plot(sol.t, sol.y[0], label='U sin ruido', color=
'black', linewidth=2)
plt.legend()
plt.title('Trayectoria de Uranio con A=1000')
plt.yscale('log')
plt.xscale('log')
plt.savefig('Taller_5/2.b.pdf')


#-------------------------------
# Parte C: simulación exacta (Gillespie SSA)
#-------------------------------
# Parámetros para la simulación
dt_sde = 1e-3  # paso de tiempo pequeño para buena precisión
tmax_sde = 30.0
n_traj = 5
U0 = 10.0



def gillespie_ssa(y0_int, tmax, A, lambda_U, lambda_Np, B):
    # R: cambios de estado por reacción (filas = reacciones, columnas = U,Np,Pu)
    R = np.array([[1, 0, 0],
                  [-1, 1, 0],
                  [0, -1, 1],
                  [0, 0, -1]], dtype=int)

    t = 0.0
    U, Np, Pu = int(y0_int[0]), int(y0_int[1]), int(y0_int[2])
    times = [t]
    states = [np.array([U, Np, Pu], dtype=int)]

    while t < tmax:
        # tasas actuales para cada reacción
        rates = np.array([A, lambda_U * U, lambda_Np * Np, B * Pu], dtype=float)
        total = rates.sum()
        if total <= 0.0:
            break                                   # no hay reacciones posibles

        tau = np.random.exponential(1.0 / total)   # tiempo hasta siguiente reacción
        t += tau
        if t > tmax:
            break                                   # no aplico reacción si supero tmax

        # elijo reacción proporcional a sus tasas
        r = np.random.random() * total
        reaction_index = int(np.searchsorted(np.cumsum(rates), r))

        # aplico la reacción
        change = R[reaction_index]
        U += int(change[0]); Np += int(change[1]); Pu += int(change[2])
        U = max(U, 0); Np = max(Np, 0); Pu = max(Pu, 0)  # seguridad contra negativos

        times.append(t)
        states.append(np.array([U, Np, Pu], dtype=int))

    times = np.array(times)
    states = np.array(states).T                     # forma (3, Npoints)
    return times, states

# Genero y dibujo varias trayectorias Gillespie SSA (discretas) comparadas con determinista
n_ssa = 5
plt.figure(figsize=(8, 5))
plt.plot(t, U, color='C0', label='Determinista U', linewidth=2)
plt.plot(t, Np, color='C1', label='Determinista Np', linewidth=2)
plt.plot(t, Pu, color='C2', label='Determinista Pu', linewidth=2)

for j in range(n_ssa):
    times_ssa, states_ssa = gillespie_ssa([10, 10, 10], tmax_sde, A, lambda_U, lambda_Np, B)
    # dibujo como escalones para mostrar naturaleza discreta
    plt.step(times_ssa, states_ssa[0, :], where='post', color='C0', alpha=0.6)
    plt.step(times_ssa, states_ssa[1, :], where='post', color='C1', alpha=0.6)
    plt.step(times_ssa, states_ssa[2, :], where='post', color='C2', alpha=0.6)

plt.xlabel('Tiempo (días)')
plt.ylabel('Cantidad (unidades)')
plt.xscale('log') # escala logarítmica en x para ver mejor el inicio
plt.yscale('log') # escala logarítmica en y para ver mejor el inicio
plt.title('Trayectorias Gillespie SSA (discretas) vs determinista')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Taller_5/2.c.pdf', dpi=300)



# -------------------------------
# Parte D: Probabilidad de concentración crítica (Pu >= 80) en 30 días
# -------------------------------
# parámetros del experimento Monte Carlo
Nsim = 1000               # número de trayectorias (≈1000 como pide el enunciado)
Pu_crit = 80.0            # umbral crítico de plutonio
tmax = tmax_sde           # tiempo máximo (reuso de la variable definida en la Parte B/C)

# ---------------------------------------------------------------------
# Me aseguro de tener disponible la función vectorial sde_rk2_system.
# Si no existe en el archivo actual la defino aquí (versión compacta).
# ---------------------------------------------------------------------
try:
    sde_rk2_system  # pruebo si existe
except NameError:
    # Defino la versión vectorial RK2 estocástica que retorna (t, sol) con sol.shape == (3, len(t))
    def sde_rk2_system(y0, tmax_local, dt_local, A_local, lambda_U_local, lambda_Np_local, B_local):
        t_sde = np.arange(0.0, tmax_local + dt_local, dt_local)
        sol = np.zeros((3, t_sde.size))
        sol[:, 0] = np.array(y0, dtype=float).copy()

        def mu_vec(y):
            U_, Np_, Pu_ = y
            return np.array([A_local - lambda_U_local * U_,
                             lambda_U_local * U_ - lambda_Np_local * Np_,
                             lambda_Np_local * Np_ - B_local * Pu_], dtype=float)

        def sigma_vec(y):
            U_, Np_, Pu_ = y
            sU = np.sqrt(max(A_local + lambda_U_local * max(U_, 0.0), 0.0))
            sNp = np.sqrt(max(lambda_U_local * max(U_, 0.0) + lambda_Np_local * max(Np_, 0.0), 0.0))
            sPu = np.sqrt(max(lambda_Np_local * max(Np_, 0.0) + B_local * max(Pu_, 0.0), 0.0))
            return np.array([sU, sNp, sPu], dtype=float)

        for j in range(1, t_sde.size):
            y = sol[:, j-1].copy()
            m = mu_vec(y)
            s = sigma_vec(y)
            # usado el mismo W y S para K1 y K2 según enunciado (reduce varianza del integrador)
            W = np.random.normal(0.0, 1.0, size=3)
            S = np.random.choice([-1.0, 1.0], size=3)
            K1 = dt_local * m + (W - S) * np.sqrt(dt_local) * s
            m2 = mu_vec(y + K1)
            s2 = sigma_vec(y + K1)
            K2 = dt_local * m2 + (W + S) * np.sqrt(dt_local) * s2
            y_new = y + 0.5 * (K1 + K2)
            sol[:, j] = np.maximum(y_new, 0.0)
        return t_sde, sol

# ---------------------------------------------------------------------
# Método A: estimación por Gillespie SSA (llamo a gillespie_ssa definido en la Parte C)
# ---------------------------------------------------------------------
k_ssa = 0
for i in range(Nsim):
    times_ssa, states_ssa = gillespie_ssa([10, 10, 10], tmax, A, lambda_U, lambda_Np, B)
    Pu_traj = states_ssa[2, :]
    if np.any(Pu_traj >= Pu_crit):
        k_ssa += 1

p_ssa = k_ssa / Nsim
se_ssa = np.sqrt(p_ssa * (1.0 - p_ssa) / Nsim).
ci_freq_ssa = (max(0.0, p_ssa - se_ssa), min(1.0, p_ssa + se_ssa))

from scipy.stats import beta
post_a_ssa = 1 + k_ssa
post_b_ssa = 1 + Nsim - k_ssa
ci_bayes_ssa = (beta.ppf(0.025, post_a_ssa, post_b_ssa), beta.ppf(0.975, post_a_ssa, post_b_ssa))
p_bayes_mean_ssa = beta.mean(post_a_ssa, post_b_ssa)

# ---------------------------------------------------------------------
# Método B: estimación por aproximación SDE (vectorial). Uso sde_rk2_system definido más arriba.
# ---------------------------------------------------------------------
dt_mc = 1e-5
k_sde = 0
for i in range(Nsim):
    t_sim, sol_sim = sde_rk2_system([10.0, 10.0, 10.0], tmax, dt_mc, A, lambda_U, lambda_Np, B)
    Pu_sim = sol_sim[2, :]
    if np.any(Pu_sim >= Pu_crit):
        k_sde += 1

p_sde = k_sde / Nsim
se_sde = np.sqrt(p_sde * (1.0 - p_sde) / Nsim)
ci_freq_sde = (max(0.0, p_sde - se_sde), min(1.0, p_sde + se_sde))

post_a_sde = 1 + k_sde
post_b_sde = 1 + Nsim - k_sde
ci_bayes_sde = (beta.ppf(0.025, post_a_sde, post_b_sde), beta.ppf(0.975, post_a_sde, post_b_sde))
p_bayes_mean_sde = beta.mean(post_a_sde, post_b_sde)

# ---------------------------------------------------------------------
# Guardado de resultados en 2.d.txt (porcentajes) y salida breve en consola
# ---------------------------------------------------------------------
with open('Taller_5/2.d.txt', 'w') as fh:
    fh.write('Resultados 2.d - Probabilidad de Pu >= 80 en 30 dias\n')
    fh.write(f'Nsim = {Nsim}\n\n')

    fh.write('Gillespie SSA:\n')
    fh.write(f'  k = {k_ssa} / {Nsim}\n')
    fh.write(f'  p_hat (freq) = {100.0*p_ssa:.4f} %\n')
    fh.write(f'  CI freq = [{100.0*ci_freq_ssa[0]:.4f} %, {100.0*ci_freq_ssa[1]:.4f} %]\n')
    fh.write(f'  Posterior Beta mean = {100.0*p_bayes_mean_ssa:.4f} %\n')
    fh.write(f'  Credible 95% (beta) = [{100.0*ci_bayes_ssa[0]:.4f} %, {100.0*ci_bayes_ssa[1]:.4f} %]\n\n')

    fh.write('SDE RK2 vectorial (aprox):\n')
    fh.write(f'  k = {k_sde} / {Nsim}\n')
    fh.write(f'  p_hat (freq) = {100.0*p_sde:.4f} %\n')
    fh.write(f'  CI freq = [{100.0*ci_freq_sde[0]:.4f} %, {100.0*ci_freq_sde[1]:.4f} %]\n')
    fh.write(f'  Posterior Beta mean = {100.0*p_bayes_mean_sde:.4f} %\n')
    fh.write(f'  Credible 95% (beta) = [{100.0*ci_bayes_sde[0]:.4f} %, {100.0*ci_bayes_sde[1]:.4f} %]\n\n')

    fh.write('Discusion breve:\n')
    fh.write('  La estimacion SSA es la referencia exacta a nivel de eventos discretos; la aproximacion SDE continua\n')
    fh.write('  puede sub/ sobreestimar la probabilidad dependiendo de dt y de la aproximacion de ruido.\n')

print('2.d: Guardado 2.d.txt con resultados.')
print(f'Gillespie: k={k_ssa}, p={p_ssa:.5f}, CI_freq=({ci_freq_ssa[0]:.5f},{ci_freq_ssa[1]:.5f}), CI_bayes=({ci_bayes_ssa[0]:.5f},{ci_bayes_ssa[1]:.5f})')
print(f'SDE     : k={k_sde}, p={p_sde:.5f}, CI_freq=({ci_freq_sde[0]:.5f},{ci_freq_sde[1]:.5f}), CI_bayes=({ci_bayes_sde[0]:.5f},{ci_bayes_sde[1]:.5f})')

