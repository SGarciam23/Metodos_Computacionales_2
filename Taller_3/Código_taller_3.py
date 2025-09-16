import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --------------------------------------------------
# 1.a Sistema depredador-presa (Lotka–Volterra)
# --------------------------------------------------
def lotka_volterra(t, z, alpha=2, beta=1.5, gamma=0.3, delta=0.4):
    x, y = z
    dxdt = alpha*x - beta*x*y
    dydt = -gamma*y + delta*x*y
    return [dxdt, dydt]

def conserved_lv(x, y, alpha=2, beta=1.5, gamma=0.3, delta=0.4):
    return delta*x - gamma*np.log(x) + beta*y - alpha*np.log(y)

def simulate_lotka_volterra():
    t_span = (0, 50)
    t_eval = np.linspace(*t_span, 2000)
    sol = solve_ivp(lotka_volterra, t_span, [3, 2], t_eval=t_eval, rtol=1e-9, atol=1e-9)
    x, y = sol.y
    V = conserved_lv(x, y)

    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    axs[0].plot(sol.t, x, label="Presas (x)")
    axs[0].plot(sol.t, y, label="Depredadores (y)")
    axs[0].set_ylabel("Población")
    axs[0].legend()

    axs[1].plot(x, y)
    axs[1].set_xlabel("x (presas)")
    axs[1].set_ylabel("y (depredadores)")

    axs[2].plot(sol.t, V)
    axs[2].set_xlabel("Tiempo")
    axs[2].set_ylabel("Cantidad conservada V")

    fig.tight_layout()
    plt.savefig("1.a.pdf")
    plt.close()


# --------------------------------------------------
# 1.b Problema de Landau
# --------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def landau(t, z, q=7.5284, B0=0.438, E0=0.7423, m=3.8428, k=1.0014):
    """
    Ecuaciones de movimiento para el problema de Landau.
    z = [x, y, vx, vy]
    """
    x, y, vx, vy = z
    ax = (q*E0*(np.sin(k*x) + k*x*np.cos(k*x)) + q*B0*vy) / m
    ay = (-q*B0*vx) / m
    return [vx, vy, ax, ay]

def conserved_landau(x, y, vx, vy, q=7.5284, B0=0.438, E0=0.7423, m=3.8428, k=1.0014):
    """
    Cantidades conservadas:
    - Momento conjugado Πy
    - Energía total
    """
    Piy = m*vy - q*B0*x
    energy = 0.5*m*(vx**2 + vy**2) - q*E0*x*np.sin(k*x)
    return Piy, energy

def simulate_landau():
    # intervalo temporal y malla
    t_span = (0, 30)
    t_eval = np.linspace(*t_span, 3000)

    # condición inicial [x, y, vx, vy]
    z0 = [0, 0, 1, 0]

    # resolver usando un integrador de alta precisión
    sol = solve_ivp(
        landau, t_span, z0, t_eval=t_eval,
        rtol=1e-9, atol=1e-9,
        method="DOP853", max_step=0.01
    )

    x, y, vx, vy = sol.y
    Piy, E = conserved_landau(x, y, vx, vy)

    # energía relativa para que se vea plana
    E_rel = (E - E[0]) / E[0]

    # gráficas
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))

    # trayectoria
    axs[0].plot(x, y)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("Trayectoria en el plano (x,y)")

    # momento conjugado
    axs[1].plot(sol.t, Piy)
    axs[1].set_ylabel("Momento conjugado Πy")
    axs[1].set_title("Conservación de Πy")

    # energía relativa
    axs[2].plot(sol.t, E_rel)
    axs[2].set_xlabel("Tiempo")
    axs[2].set_ylabel("ΔE / E0")
    axs[2].set_title("Conservación de la energía (relativa)")

    fig.tight_layout()
    plt.savefig("1.b.pdf")
    plt.close()


# --------------------------------------------------
# 1.c Sistema binario (gravedad)
# --------------------------------------------------
def binary_system(t, z, G=1, m=1.7):
    r1x, r1y, r2x, r2y, v1x, v1y, v2x, v2y = z
    dx = r2x - r1x
    dy = r2y - r1y
    r = np.sqrt(dx**2 + dy**2)
    F = G*m*m / r**3
    a1x, a1y = F*dx/m, F*dy/m
    a2x, a2y = -F*dx/m, -F*dy/m
    return [v1x, v1y, v2x, v2y, a1x, a1y, a2x, a2y]

def conserved_binary(r1, r2, v1, v2, G=1, m=1.7):
    dx, dy = r2 - r1
    r = np.sqrt(dx**2 + dy**2)
    v1sq, v2sq = np.dot(v1, v1), np.dot(v2, v2)
    E = 0.5*m*(v1sq+v2sq) - G*m*m/r
    L = m*(np.cross(r1, v1) + np.cross(r2, v2))
    return E, L

def simulate_binary():
    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 2000)
    z0 = [0, 0, 1, 1, 0, 0.5, 0, -0.5]
    sol = solve_ivp(binary_system, t_span, z0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    r1 = sol.y[0:2].T
    r2 = sol.y[2:4].T
    v1 = sol.y[4:6].T
    v2 = sol.y[6:8].T

    E, L = [], []
    for i in range(len(sol.t)):
        e, l = conserved_binary(r1[i], r2[i], v1[i], v2[i])
        E.append(e); L.append(l)

    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    axs[0].plot(r1[:,0], r1[:,1], label="Estrella 1")
    axs[0].plot(r2[:,0], r2[:,1], label="Estrella 2")
    axs[0].legend()

    axs[1].plot(sol.t, E)
    axs[1].set_ylabel("Energía total")

    axs[2].plot(sol.t, L)
    axs[2].set_xlabel("Tiempo")
    axs[2].set_ylabel("Momento angular total")

    fig.tight_layout()
    plt.savefig("1.c.pdf")
    plt.close()


# --------------------------------------------------
# Ejecutar todo el punto 1
# --------------------------------------------------
if __name__ == "__main__":
    simulate_lotka_volterra()
    simulate_landau()
    simulate_binary()


# -----------------------------
# PUNTO 2
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ======================================
# Constantes del problema
# ======================================
g = 9.773   # Gravedad en Bogotá (m/s^2)
m = 10      # Masa del proyectil (kg)

# Parámetros empíricos de β(y)
A, B, C = 1.642, 40.624, 2.36

# ======================================
# Definición de β(y) coeficiente de resistencia del aire
# ======================================
def beta_y(y):
    return A * (1 - y / B) ** C if y < B else 0.0

# ======================================
# Ecuaciones de movimiento
# ======================================
def equations(t, Y):
    x, vx, y, vy = Y
    v = np.sqrt(vx*2 + vy*2)
    beta = beta_y(y)
    ax = -beta * v * vx / m
    ay = -g - (beta * v * vy / m)
    return [vx, ax, vy, ay]

# ======================================
# Evento: detener integración al tocar el suelo
# ======================================
def hit_ground(t, Y):
    return Y[2]
hit_ground.terminal = True
hit_ground.direction = -1

# ======================================
# Simulación de trayectoria
# ======================================
def simulate(v0, theta):
    v0x, v0y = v0 * np.cos(theta), v0 * np.sin(theta)
    Y0 = [0, v0x, 0, v0y]
    sol = solve_ivp(equations, [0, 100], Y0,
                    method="RK45", max_step=0.01,
                    events=hit_ground)
    return sol.t, sol.y

# ======================================
# 2.a Alcance máximo vs v0 (con θ = 45°)
# ======================================
v0_values = np.linspace(10, 140, 30)
ranges = []
for v0 in v0_values:
    _, Y = simulate(v0, np.pi/4)
    ranges.append(Y[0, -1])

plt.figure(figsize=(8,5))
plt.plot(v0_values, ranges, marker="o")
plt.xlabel(r"$v_0$ (m/s)")
plt.ylabel(r"$x_{max}$ (m)")
plt.title("Alcance máximo vs $v_0$ con $\\theta=45°$")
plt.grid()
plt.savefig("2.a.pdf")
plt.close()

# ======================================
# 2.b Función para acertar a un objetivo
# ======================================
def angle_to_hit_target(v0, target_x, target_y, tol=0.5):
    thetas = np.linspace(0.01, np.pi/2 - 0.01, 200)
    solutions = []
    for theta in thetas:
        _, Y = simulate(v0, theta)
        x_vals, y_vals = Y[0], Y[2]
        # Buscar si pasa cerca del objetivo
        dist = np.sqrt((x_vals - target_x)*2 + (y_vals - target_y)*2)
        if np.min(dist) < tol:
            solutions.append(theta)
    return solutions

# Ejemplo: objetivo en (12, 0)
solutions = angle_to_hit_target(20, 12, 0)
print("Ángulos que pegan en (12,0) con v0=20 m/s:", np.degrees(solutions))

# ======================================
# 2.c Varias opciones (pares v0, theta)
# ======================================
def find_solutions(target_x, target_y):
    v0_vals = np.linspace(10, 140, 20)
    theta_vals = np.linspace(0.01, np.pi/2 - 0.01, 50)
    sols = []
    for v0 in v0_vals:
        for theta in theta_vals:
            _, Y = simulate(v0, theta)
            x_vals, y_vals = Y[0], Y[2]
            dist = np.sqrt((x_vals - target_x)*2 + (y_vals - target_y)*2)
            if np.min(dist) < 0.5:
                sols.append((v0, theta))
    return sols

target_x, target_y = 12, 0
solutions_v0_theta = find_solutions(target_x, target_y)

# Gráfica de soluciones
plt.figure(figsize=(8,5))
for v0, theta in solutions_v0_theta:
    plt.scatter(v0, np.degrees(theta), color="blue")
plt.xlabel(r"$v_0$ (m/s)")
plt.ylabel(r"$\theta_0$ (grados)")
plt.title(f"Condiciones iniciales que dan en el blanco ({target_x},{target_y})")
plt.grid()
plt.savefig("2.c.pdf")
plt.close()

# -----------------------------
# PUNTO 3
# -----------------------------
# Código optimizado — manteniendo el mismo método (shooting + solve_ivp + brentq)
# Cambios: memoización, prints de monitoreo, paralelización opcional, menor ngrid por defecto.
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.optimize import brentq
from functools import lru_cache
import concurrent.futures
import os

# -----------------------------
# Parámetros (usar los indicados)
# -----------------------------
hbar = 0.1
m = 1.0
a = 0.8
x0 = 10.0

# dominio global para buscar raíces de V(x)-E
XMIN_GLOBAL = x0 - 8.0
XMAX_GLOBAL = x0 + 12.0

# paso espacial máximo pedido
DX = 0.01
MAX_STEP = 0.01

# condición inicial mejorada
PSI_INIT = 1e-6

# -----------------------------
# Opciones de optimización
# -----------------------------
# Cuántos trabajadores para el escaneo paralelo. 1 = sin paralelizar.
# Ajuste según CPU; por defecto usa todos los CPUs disponibles.
N_WORKERS = max(1, os.cpu_count() or 1)

# Cada cuántas energías imprime progreso (cuando no está paralelizado)
PROGRESS_EVERY = 500

# -----------------------------
# Potencial de Morse
# -----------------------------
def V(x):
    return (1.0 - np.exp(-a * (x0 - x)))**2 - 1.0

# -----------------------------
# Ecuaciones de Schrödinger
# y' = f(x,y;E) con y=[psi, psi']
# -----------------------------
def schrodinger(x, y, E):
    psi, phi = y
    dpsi = phi
    dphi = (2.0 * m / hbar**2) * (V(x) - E) * psi
    return [dpsi, dphi]

# -----------------------------
# Encontrar puntos de giro
# Uso de cache para energías muy similares (redondeadas)
# -----------------------------
@lru_cache(maxsize=1024)
def turning_points_cached(E_key, xmin=XMIN_GLOBAL, xmax=XMAX_GLOBAL, ngrid=5000):
    """
    E_key: entero que representa E*1e12 (clave cacheable).
    Esta función es la versión cacheable; se llama dentro de turning_points().
    """
    E = E_key / 1e12
    xs = np.linspace(xmin, xmax, ngrid)
    f = V(xs) - E
    sign_changes = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    roots = []
    for idx in sign_changes:
        xL, xR = xs[idx], xs[idx+1]
        try:
            root = brentq(lambda xx: V(xx)-E, xL, xR, xtol=1e-12, maxiter=100)
            roots.append(root)
        except Exception:
            pass
    # deduplicate y redondeo a 10 decimales para estabilidad
    roots = np.array(sorted(list(set([round(r,10) for r in roots]))))
    return tuple(roots.tolist())

def turning_points(E, xmin=XMIN_GLOBAL, xmax=XMAX_GLOBAL, ngrid=5000):
    # redondeo para clave de cache
    E_key = int(round(E * 1e12))
    return list(turning_points_cached(E_key, xmin, xmax, ngrid))

# -----------------------------
# Integrar ODE para energía dada
# -----------------------------
def integrate_for_energy(E, x_left, x_right, dx=DX):
    # Creamos el grid de evaluación con paso dx
    x_eval = np.arange(x_left, x_right + dx, dx)
    y0 = [PSI_INIT, 0.0] # psi' = 0.0 en el punto de retorno (igual que en su versión)
    sol = solve_ivp(fun=schrodinger, t_span=(x_eval[0], x_eval[-1]), y0=y0,
                    t_eval=x_eval, args=(E,), max_step=MAX_STEP, method='RK45')
    return sol.t, sol.y[0]

# -----------------------------
# Función para shooting (cacheada)
# -----------------------------
@lru_cache(maxsize=4096)
def psi_at_right_cached(E_key):
    """
    E_key es E*1e12 (entero). Retorna psi_right (float) o np.nan
    """
    E = E_key / 1e12
    tps = turning_points(E)
    if len(tps) < 2:
        return np.nan
    x1, x2 = tps[0], tps[1]
    # empiezo ligeramente después del turning point izquierdo para evitar singularidades
    xL = x1 - 0.02
    xR = x2 + 0.02
    try:
        xs, psi = integrate_for_energy(E, xL, xR)
        # si psi[0] es 0 (numéricamente improbable con PSI_INIT), devolver nan
        if abs(psi[0]) < 1e-16:
            return np.nan
        psi = psi / psi[0]
        return float(psi[-1])
    except Exception:
        return np.nan

def psi_at_right(E):
    E_key = int(round(E * 1e12))
    return psi_at_right_cached(E_key)

# -----------------------------
# Escaneo de energías (paralelizable)
# -----------------------------
def compute_psi_vals(E_scan, parallel_workers=1):
    start = time.perf_counter()
    print(f"[INFO] Iniciando escaneo de {len(E_scan)} energías. Workers = {parallel_workers}")
    psi_vals = None
    if parallel_workers is not None and parallel_workers > 1:
        # Paraleliza usando procesos (cada proceso invocará la misma lógica)
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as ex:
            # map garantiza orden de salida igual al orden de entrada
            futures = list(ex.map(psi_at_right, E_scan))
            psi_vals = np.array(list(futures), dtype=float)
    else:
        # secuencial con prints de progreso
        psi_list = []
        for i, E in enumerate(E_scan):
            if (i % PROGRESS_EVERY) == 0:
                elapsed = time.perf_counter() - start
                print(f"[PROGRESS] E index {i}/{len(E_scan)} — E={E:.6g} — elapsed {elapsed:.2f}s")
            psi_list.append(psi_at_right(E))
        psi_vals = np.array(psi_list, dtype=float)
    total = time.perf_counter() - start
    print(f"[INFO] Escaneo completo en {total:.2f} s")
    return psi_vals

# -----------------------------
# Búsqueda de autovalores (ceros de psi_at_right)
# -----------------------------
def find_bound_states(E_scan, psi_vals):
    print("[INFO] Buscando cambios de signo para localizar intervalos candidatos...")
    found_energies = []
    for i in range(len(E_scan) - 1):
        f1, f2 = psi_vals[i], psi_vals[i+1]
        if np.isnan(f1) or np.isnan(f2):
            continue
        if f1 * f2 < 0:
            E_low, E_high = E_scan[i], E_scan[i+1]
            try:
                # brentq llamará a psi_at_right que está cacheada
                root = brentq(lambda EE: psi_at_right(EE), E_low, E_high, xtol=1e-8, maxiter=50)
                if not any(abs(root - E0) < 1e-6 for E0 in found_energies):
                    found_energies.append(root)
                    print(f"[ROOT] Encontrado autovalor E = {root:.10f} en intervalo [{E_low:.6g}, {E_high:.6g}]")
            except Exception as exc:
                # no fallamos el programa, solo saltamos
                print(f"[WARN] brentq falló en intervalo [{E_low:.6g}, {E_high:.6g}]: {exc}")
                pass
    found_energies = sorted(found_energies)
    print(f"[INFO] Total autovalores encontrados: {len(found_energies)}")
    return found_energies

# -----------------------------
# Rutina principal
# -----------------------------
def main():
    total_start = time.perf_counter()

    # Escaneo grueso (puede ajustar número de puntos)
    # REDUCIMOS ENERGIAS DE 20000 A 1000 PARA UNA VERIFICACIÓN MÁS FACIL
    E_SCAN = np.linspace(-0.9999, -0.001, 1000)

    # Compute psi values (paralelo opcional)
    psi_vals = compute_psi_vals(E_SCAN, parallel_workers=N_WORKERS)

    # Encontrar estados a partir de cambios de signo
    found_energies = find_bound_states(E_SCAN, psi_vals)

    # Integrar y normalizar cada estado encontrado (sin paralelizar por simplicidad)
    states = []
    print("[INFO] Integrando y normalizando cada estado encontrado...")
    for E in found_energies:
        tps = turning_points(E)
        if len(tps) < 2:
            continue
        x1, x2 = tps[0], tps[1]
        
        # Corrección: Integrar SÓLO dentro de los puntos de retorno
        xL = x1 - 0.02
        xR = x2 + 0.02
        xs, psi = integrate_for_energy(E, xL, xR)

        norm = np.sqrt(integrate.simpson(psi**2, xs))
        if norm == 0 or np.isnan(norm):
            print(f"[WARN] Norm problem for E={E:.10f}: norm={norm}")
            continue

        psi_n = psi / norm
        states.append({'E': E, 'x': xs, 'psi': psi_n, 'x1': x1, 'x2': x2})
        print(f"[STATE] Estado E={E:.10f} integrado y normalizado (len(xs)={len(xs)})")

    # Plots (igual que su version)
    x_global = np.linspace(XMIN_GLOBAL, XMAX_GLOBAL, 4000)
    V_global = V(x_global)

    plt.figure(figsize=(10,6))
    plt.plot(x_global, V_global, linewidth=1.2, label="Potencial de Morse")

    amp = 0.02
    colors = plt.cm.viridis(np.linspace(0,1,len(states)))

    for n, st in enumerate(states):
        E = st['E']
        xs = st['x']
        psi = st['psi']
        
        # Para que no se vea una línea recta, usamos el dominio completo para el ploteo
        # pero rellenamos con ceros fuera del rango de integración
        x_plot = np.linspace(XMIN_GLOBAL, XMAX_GLOBAL, 1000)
        psi_plot = np.zeros_like(x_plot)
        
        # Encontramos los índices correspondientes a los puntos de retorno para ploteo
        idx_start = np.searchsorted(x_plot, xs[0])
        idx_end = np.searchsorted(x_plot, xs[-1])
        
        # Interpolamos la psi calculada en el grid de ploteo
        psi_interpolated = np.interp(x_plot[idx_start:idx_end], xs, psi)
        psi_plot[idx_start:idx_end] = psi_interpolated
        
        plt.plot(x_plot, psi_plot*amp + E, lw=1, color=colors[n], label=f"n={n}")
        plt.hlines(E, xs[0], xs[-1], colors="gray", linestyles='--', lw=0.6)

    plt.xlabel("x")
    plt.ylabel("Energía")
    plt.title("Estados ligados en el potencial de Morse (corregido)")
    plt.ylim(-1.2, 0.2)
    plt.xlim(XMIN_GLOBAL, XMAX_GLOBAL)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    '''# Gráfica de la función de shooting (opcional)
    E_SCAN_SHOOTING = np.linspace(-1.0, -0.01, 2000)
    psi_vals_shooting = compute_psi_vals(E_SCAN_SHOOTING, parallel_workers=1)  # para la gráfica, uso secuencial
    plt.figure(figsize=(7,5))
    plt.plot(E_SCAN_SHOOTING, psi_vals_shooting, "-")
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("Energía")
    plt.ylabel("psi_at_right(E)")
    plt.title("Función de shooting para localizar autovalores")
    # límites adaptativos si hay valores grandes
    finite_vals = psi_vals_shooting[np.isfinite(psi_vals_shooting)]
    if finite_vals.size > 0:
        vmin, vmax = np.percentile(finite_vals, [1,99])
        rng = max(1.0, max(abs(vmin), abs(vmax)))
        plt.ylim(-rng, rng)
    else:
        plt.ylim(-5,5)
    plt.grid(True, alpha=0.3)
    plt.show()'''

    total_end = time.perf_counter()
    print(f"[DONE] Tiempo total (script): {total_end - total_start:.2f} s")

if __name__ == "__main__":
    print("[START] Ejecutando script optimizado para shooting + solve_ivp")
    print(f"[CONFIG] N_WORKERS={N_WORKERS}, DX={DX}, MAX_STEP={MAX_STEP}")
    main()

# -----------------------------
# PUNTO 4
# -----------------------------

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parámetros
alpha = 1.0   # mg/kl
t_max = 1e4

# Condiciones iniciales
r0 = 1.0
theta0 = np.pi / 2
dr0 = 0.0
dtheta0 = 0.0
y0 = [r0, dr0, theta0, dtheta0]

# Ecuaciones de movimiento
def elastic_pendulum(t, y):
  r, dr, theta, dtheta = y
  d2r = r * dtheta**2 - alpha * (r - 1)
  d2theta = - (2 * dr * dtheta) / r - np.sin(theta) / r
  return [dr, d2r, dtheta, d2theta]

# Evento: cruce por el eje vertical hacia abajo (y < 0)
def crossing_event(t, y):
  r, dr, theta, dtheta = y
  return np.cos(theta)  # cruce cuando cos(theta) = 0

crossing_event.terminal = False
crossing_event.direction = 0  # solo cuando pasa hacia abajo

# Integración
sol = solve_ivp(
    elastic_pendulum,
    [0, t_max],
    y0,
    method='DOP853',
    max_step=0.1,
    events=crossing_event
)

# Extraer puntos de Poincaré
r_vals = []
Pr_vals = []  # P_r = dr

for state in sol.y_events[0]:
  r, dr, theta, dtheta = state
  if np.cos(theta) > 0: # y < 0
    r_vals.append(r)
    Pr_vals.append(dr)

# Graficar P_r vs r y guardar en PDF
plt.figure(figsize=(6, 6))
plt.scatter(r_vals, Pr_vals, s=2, color='blue')
plt.xlabel(r"$r$")
plt.ylabel(r"$P_r$")
plt.title("Diagrama de Poincaré: $P_r$ vs $r$")
plt.grid(True)
plt.tight_layout()
plt.savefig("4.pdf")  # Guarda en PDF
plt.close()


