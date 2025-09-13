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
# Definición de β(y)
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.optimize import brentq

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

# escaneo de energías más fino
E_SCAN = np.linspace(-0.999, -0.01, 5000)

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
# -----------------------------
def turning_points(E, xmin=XMIN_GLOBAL, xmax=XMAX_GLOBAL, ngrid=20000):
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
    roots = np.array(sorted(list(set([round(r,10) for r in roots]))))
    return list(roots)

# -----------------------------
# Integrar ODE para energía dada
# -----------------------------
def integrate_for_energy(E, x_left, x_right, dx=DX):
    x_eval = np.arange(x_left, x_right + dx, dx)
    y0 = [PSI_INIT, 0.0]  # arranque mejorado
    sol = solve_ivp(fun=schrodinger, t_span=(x_left, x_right), y0=y0,
                    t_eval=x_eval, args=(E,), max_step=MAX_STEP, method='RK45')
    return sol.t, sol.y[0]

# -----------------------------
# Función para shooting
# -----------------------------
def psi_at_right(E):
    tps = turning_points(E)
    if len(tps) < 2:
        return np.nan
    x1, x2 = tps[0], tps[1]
    xL = x1 - 2.0
    xR = x2 + 1.0
    try:
        xs, psi = integrate_for_energy(E, xL, xR)
        psi = psi / np.max(np.abs(psi))  # normalizar para evitar overflow
        return psi[-1]
    except Exception:
        return np.nan

# -----------------------------
# Escaneo de energías
# -----------------------------
found_energies = []
psi_vals = []
for E in E_SCAN:
    psi_vals.append(psi_at_right(E))
psi_vals = np.array(psi_vals)

for i in range(len(E_SCAN)-1):
    f1, f2 = psi_vals[i], psi_vals[i+1]
    if np.isnan(f1) or np.isnan(f2):
        continue
    if f1 * f2 < 0:
        E_low, E_high = E_SCAN[i], E_SCAN[i+1]
        try:
            root = brentq(lambda EE: psi_at_right(EE), E_low, E_high, xtol=1e-8, maxiter=50)
            if not any(abs(root - E0) < 1e-6 for E0 in found_energies):
                found_energies.append(root)
        except Exception:
            pass

found_energies = sorted(found_energies)

# -----------------------------
# Integrar y normalizar cada estado
# -----------------------------
states = []
for E in found_energies:
    tps = turning_points(E)
    if len(tps) < 2:
        continue
    x1, x2 = tps[0], tps[1]
    xL = x1 - 2.0
    xR = x2 + 1.0
    xs, psi = integrate_for_energy(E, xL, xR)
    norm = np.sqrt(integrate.simpson(psi**2, xs))
    if norm == 0 or np.isnan(norm):
        continue
    psi_n = psi / norm
    states.append({'E': E, 'x': xs, 'psi': psi_n, 'x1': x1, 'x2': x2})

# -----------------------------
# Guardar energías
# -----------------------------
with open("3_energies.txt", "w") as f:
    f.write("n\tE_numeric\n")
    for i, st in enumerate(states):
        f.write(f"{i}\t{st['E']:.10f}\n")

print(f"Encontradas {len(states)} energías ligadas (listadas en 3_energies.txt).")

# -----------------------------
# Gráfica final
# -----------------------------
x_global = np.linspace(XMIN_GLOBAL, XMAX_GLOBAL, 4000)
V_global = V(x_global)

plt.figure(figsize=(8,6))
plt.plot(x_global, V_global, color='black', linewidth=1.2, label="Potencial de Morse")

amp = 0.15
for n, st in enumerate(states):
    E = st['E']
    xs = st['x']
    psi = st['psi']
    plt.plot(xs, psi*amp + E, lw=1, label=f"n={n}")
    plt.hlines(E, xs[0], xs[-1], colors="gray", linestyles='--', lw=0.6)
    plt.scatter([st['x1'], st['x2']], [E, E], s=10, color="red")

plt.xlabel("x")
plt.ylabel("Energía / ψ(x)")
plt.title("Estados ligados en el potencial de Morse")
plt.ylim(-1.2, 0.2)
plt.xlim(XMIN_GLOBAL, XMAX_GLOBAL)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
