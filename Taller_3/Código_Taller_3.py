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

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.optimize import brentq
from functools import lru_cache

# -----------------------------
# Parámetros
# -----------------------------
hbar = 0.1
m = 1.0
a = 0.8
x0 = 10.0

XMIN_GLOBAL = 0.0
XMAX_GLOBAL = 20.0

DX = 0.01
MAX_STEP = 0.01

# valores por defecto (se usan si no hay región prohibida)
PSI_INIT = 1e-6
PSI_DERIV_INIT = 1e-6

N_WORKERS = 1
PROGRESS_EVERY = 500

# -----------------------------
# Potencial de Morse
# -----------------------------
def V(x):
    return (1.0 - np.exp(a * (x - x0)))**2 - 1.0

# -----------------------------
# Ecuaciones de Schrödinger
# -----------------------------
def schrodinger(x, y, E):
    psi, phi = y
    dpsi = phi
    dphi = (V(x) - E) * psi / (hbar**2)
    return [dpsi, dphi]

# -----------------------------
# Puntos de giro
# -----------------------------
@lru_cache(maxsize=1024)
def turning_points_cached(E_key, xmin=XMIN_GLOBAL, xmax=XMAX_GLOBAL, ngrid=2000):
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
    roots = sorted(set([round(r,10) for r in roots]))
    return tuple(roots)

def turning_points(E, xmin=XMIN_GLOBAL, xmax=XMAX_GLOBAL, ngrid=2000):
    E_key = int(round(E * 1e12))
    return list(turning_points_cached(E_key, xmin, xmax, ngrid))

# -----------------------------
# Integración con condiciones de frontera suaves
# -----------------------------
def integrate_for_energy(E, x_left, x_right, dx=DX):
    x_eval = np.arange(x_left, x_right + dx, dx)

    # condiciones iniciales desde decaimiento exponencial
    Vleft = V(x_left)
    if Vleft > E:
        kappa = np.sqrt(2 * m * (Vleft - E)) / hbar
        psi0 = np.exp(-kappa * (turning_points(E)[0] - x_left))
        dpsi0 = kappa * psi0
    else:
        psi0, dpsi0 = PSI_INIT, PSI_DERIV_INIT

    y0 = [psi0, dpsi0]

    sol = solve_ivp(fun=schrodinger, t_span=(x_eval[0], x_eval[-1]), y0=y0,
                    t_eval=x_eval, args=(E,), max_step=MAX_STEP, method='RK45')
    return sol.t, sol.y[0]

# -----------------------------
# Shooting
# -----------------------------
@lru_cache(maxsize=8192)
def psi_at_right_cached(E_key):
    E = E_key / 1e12
    tps = turning_points(E)
    if len(tps) < 2:
        return np.nan

    x1, x2 = tps[0], tps[1]
    xL = x1 - 2.0
    xR = x2 + 0.5

    try:
        xs, psi = integrate_for_energy(E, xL, xR)
        psi = psi / psi[0]  # normalización relativa
        return float(psi[-1])
    except Exception:
        return np.nan

def psi_at_right(E):
    E_key = int(round(E * 1e12))
    return psi_at_right_cached(E_key)

# -----------------------------
# Escaneo de energías
# -----------------------------
def compute_psi_vals(E_scan, parallel_workers=1):
    start = time.perf_counter()
    print(f"[INFO] Iniciando escaneo de {len(E_scan)} energías...")
    psi_list = []
    for i, E in enumerate(E_scan):
        if (i % PROGRESS_EVERY) == 0:
            elapsed = time.perf_counter() - start
            print(f"[PROGRESS] E index {i}/{len(E_scan)} — E={E:.6g} — elapsed {elapsed:.2f}s")
        psi_list.append(psi_at_right(E))
    psi_vals = np.array(psi_list, dtype=float)
    print(f"[INFO] Escaneo completo en {time.perf_counter()-start:.2f} s")
    return psi_vals

# -----------------------------
# Localizar autovalores
# -----------------------------
def find_bound_states(E_scan, psi_vals):
    print("[INFO] Buscando autovalores...")
    found_energies = []
    for i in range(len(E_scan) - 1):
        f1, f2 = psi_vals[i], psi_vals[i+1]
        if np.isnan(f1) or np.isnan(f2):
            continue
        if f1 * f2 < 0:
            E_low, E_high = E_scan[i], E_scan[i+1]
            try:
                root = brentq(lambda EE: psi_at_right(EE), E_low, E_high, xtol=1e-8, maxiter=60)
                if not any(abs(root - E0) < 1e-7 for E0 in found_energies):
                    found_energies.append(root)
                    print(f"[ROOT] E = {root:.10f}")
            except Exception as exc:
                print(f"[WARN] brentq falló en [{E_low},{E_high}]: {exc}")
                pass
    return sorted(found_energies)

# -----------------------------
# Rutina principal
# -----------------------------
def main():
    total_start = time.perf_counter()

    E_SCAN = np.linspace(-0.9999, -0.001, 1000)
    psi_vals = compute_psi_vals(E_SCAN, parallel_workers=N_WORKERS)
    found_energies = find_bound_states(E_SCAN, psi_vals)

    # Guardar energías en archivo TXT
    with open("energias.txt", "w") as f:
        for n, E in enumerate(found_energies):
            f.write(f"n={n}, E={E:.10f}\n")
    print("[INFO] Energías guardadas en energias.txt")

    states = []
    print("[INFO] Integrando y normalizando cada estado...")
    for E in found_energies:
        tps = turning_points(E)
        if len(tps) < 2:
            continue
        x1, x2 = tps[0], tps[1]
        xL = x1 - 2.0
        xR = x2 + 0.5
        xs, psi = integrate_for_energy(E, xL, xR)

        norm = np.sqrt(integrate.simpson(psi**2, xs))
        if norm == 0 or np.isnan(norm):
            continue
        psi_n = psi / norm
        states.append({'E': E, 'x': xs, 'psi': psi_n})
        print(f"[STATE] E={E:.10f} normalizado")

    # Plot final
    x_global = np.linspace(XMIN_GLOBAL, XMAX_GLOBAL, 3000)
    V_global = V(x_global)

    plt.figure(figsize=(9,6))
    plt.plot(x_global, V_global, 'k', lw=1.2, label="Potencial de Morse")

    amp = 0.05
    colors = plt.cm.viridis(np.linspace(0,1,max(1,len(states))))

    for n, st in enumerate(states):
        E = st['E']
        xs = st['x']
        psi = st['psi']

        x_plot = np.linspace(XMIN_GLOBAL, XMAX_GLOBAL, 1200)
        psi_plot = np.zeros_like(x_plot)
        idx_start = np.searchsorted(x_plot, xs[0])
        idx_end = np.searchsorted(x_plot, xs[-1])
        psi_interpolated = np.interp(x_plot[idx_start:idx_end], xs, psi)
        psi_plot[idx_start:idx_end] = psi_interpolated

        plt.plot(x_plot, psi_plot*amp + E, color=colors[n], lw=1.2, label=f"n={n}")
        plt.hlines(E, xs[0], xs[-1], colors="gray", linestyles='--', lw=0.6)

    plt.xlabel("x")
    plt.ylabel("Energía")
    plt.title("Estados ligados en el potencial de Morse")
    plt.ylim(-1.2, 0.2)
    plt.xlim(XMIN_GLOBAL, XMAX_GLOBAL)
    plt.legend()
    plt.grid(alpha=0.3)

    # Guardar la gráfica como PDF
    plt.savefig("morse_estados.pdf")
    print("[INFO] Gráfica guardada en morse_estados.pdf")

    print(f"[DONE] Tiempo total: {time.perf_counter() - total_start:.2f} s")

if __name__ == "__main__":
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

# -----------------------------
# PUNTO 7
# -----------------------------

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

def lane_emden(xi, y, n):
  theta, phi = y
  dtheta = phi
  # trata la singularidad de xi=0
  dphi = -theta**n - (2/xi)*phi if xi != 0 else 0
  return [dtheta, dphi]

def solve_lane_emden(n, xi_max=100, tol=1e-10): 
  # Condiciones cerca a xi=0 usando expansión de series
  xi0 = 1e-8
  theta0 = 1 - (n/6)*xi0**2
  phi0 = -(n/3)*xi0/2
  sol = solve_ivp(lane_emden, [xi0, xi_max], [theta0, phi0], args=(n,),
                    rtol=1e-12, atol=1e-12, max_step=0.01)

  xi_vals = sol.t
  theta_vals = sol.y[0]
  phi_vals = sol.y[1]

  # encuentra primer zero
  zero_crossings = np.where(theta_vals <= tol)[0] # Use tolerance for zero crossing
  if len(zero_crossings) == 0:
      return np.nan, np.nan, np.nan

  idx = zero_crossings[0]
  # minimo dos puntos para interpolación
  if idx == 0:
      idx = 1 

  xi1, xi2 = xi_vals[idx-1], xi_vals[idx]
  th1, th2 = theta_vals[idx-1], theta_vals[idx]

#Interpolación lineal encuentra xi_f donde theta es 0
  if abs(th2 - th1) < 1e-12:
      xi_f = xi1
  else:
      xi_f = xi1 - th1*(xi2 - xi1)/(th2 - th1)


  # deriva xi_f por interpolación lineal
  ph1 = phi_vals[idx-1]
  ph2 = phi_vals[idx]
  if abs(xi2 - xi1) < 1e-12:
      phi_f = ph1
  else:
      phi_f = ph1 + (ph2 - ph1)*(xi_f - xi1)/(xi2 - xi1)


  # masa relativa y densidad radio
  M_rel = -xi_f**2 * phi_f
  rho_ratio = xi_f / (-3 * phi_f)

  return xi_f, M_rel, rho_ratio

# Tabla
n_values = [0, 1, 1.5, 2, 3, 4, 5] # Added n=0 and n=1 back
data = []

# Especial
xi_f_0 = np.sqrt(6)
phi_f_0 = -xi_f_0/3
M_rel_0 = -xi_f_0**2 * phi_f_0
rho_ratio_0 = 1.0

# n=1
xi_f_1 = np.pi
phi_f_1 = -1/np.pi
M_rel_1 = -xi_f_1**2 * phi_f_1
rho_ratio_1 = xi_f_1 / (-3 * phi_f_1)


# n=5
xi_f_5 = np.inf
M_rel_5 = 1.0
rho_ratio_5 = np.inf


for n in n_values:
  if n == 0:
      data.append([n, xi_f_0, M_rel_0, rho_ratio_0])
  elif n == 1:
      data.append([n, xi_f_1, M_rel_1, rho_ratio_1])
  elif n == 5:
    data.append([n, xi_f_5, M_rel_5, rho_ratio_5])
  else:
    xi_f, M_rel, rho_ratio = solve_lane_emden(n)
    data.append([n, xi_f, M_rel, rho_ratio])

df = pd.DataFrame(data, columns=["Índice n", "Radio", "Masa", "rho_c / <rho>"])
print(df.to_string(index=False))
df.to_csv("7.csv", index=False)
