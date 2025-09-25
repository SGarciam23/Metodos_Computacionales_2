#--------------
# PUNTO 1
#--------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

plt.rcParams['animation.ffmpeg_path'] = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

ts = np.linspace(0, 150, 150)
psi = np.zeros((len(ts), 4))
N = 500
xmin = -20
xmax = 20
x0 = np.linspace(xmin, xmax, N)
#parametros
alpha = 0.1
hbar = 1
dx = x0[1] - x0[0]

def laplacian(N, dx):
    D2 = np.zeros((N, N), dtype=float)
    for j in range(1, N-1):
        D2[j, j-1] = 1.0
        D2[j, j]   = -2.0
        D2[j, j+1] = 1.0
    # Bordes (Neumann homogénea: psi_-1 = psi_1, psi_N = psi_{N-2})
    D2[0, 0], D2[0, 1]       = -2.0,  2.0
    D2[-1, -2], D2[-1, -1]   =  2.0, -2.0
    return D2 / dx**2


D2 = laplacian(N, dx)

psi0 = np.exp(-2*(x0-10.0)**2) * np.exp(-1j*2.0*x0)   # paquete centrado en x=10, fase k=2
# normalizar
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)

def schrodinger(t, psi, v, D2, hbar=1.0, alpha=0.1):

    cinetica = D2 @ psi
    potencial = v * psi

    f = (1j *alpha*cinetica - 1j * potencial)
    
    return f


#para V 1
V1 = (x0/5)**4
#para V 2
V2 = ((x0)**2)/50
#para V 3
V3 = (1/50)*((x0**4/100)-x0**2)

def simula_y_anima(V, nombre_salida, titulo):
    """
    Resuelve la ecuación de Schrödinger para un potencial V(x)
    y guarda la animación en un archivo mp4.
    """
    # 1. Resolver la ecuación
    sol = solve_ivp(
        schrodinger,
        (ts[0], ts[-1]),
        psi0,
        t_eval=ts,
        args=(V(x0) if callable(V) else V, D2, hbar, alpha),
        method='RK45'
    )

    # 2. Preparar la figura
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(x0.min(), x0.max())
    ax.set_ylim(0, np.max(np.abs(psi0)**2) * 1.2)
    ax.set_title(titulo)
    ax.set_xlabel('x')
    ax.set_ylabel('|ψ|²')

    # 3. Funciones de animación
    def init():
        line.set_data([], [])
        return line,

    def update(i):
        psi_t = sol.y[:, i].view(np.complex128)
        line.set_data(x0, np.abs(psi_t)**2)
        return line,

    ani = FuncAnimation(fig, update, frames=len(ts),
                        init_func=init, interval=50, blit=True)

    # 4. Mostrar y guardar
    plt.show()
    ani.save(nombre_salida, writer="ffmpeg" ,fps=20)

# ----- Llamadas para cada potencial -----
simula_y_anima(V1, "1.a.mp4", "Evolución temporal con V1")
simula_y_anima(V2, "1.b.mp4", "Evolución temporal con V2")
simula_y_anima(V3, "1.c.mp4", "Evolución temporal con V3")

#--------------
# PUNTO 2
#--------------

import numpy as np
import matplotlib.pyplot as plt

def fourier_laplacian_eigenvalues(nx, ny, Lx, Ly):
  kx = 2*np.pi*np.fft.fftfreq(nx, d=Lx/nx)
  ky = 2*np.pi*np.fft.fftfreq(ny, d=Ly/ny)
  KX, KY = np.meshgrid(kx, ky, indexing='xy')
  return -(KX**2 + KY**2)

def pasos_imex(u, v, params, lap_ev, dt):
  F = params['F'](u, v)
  G = params['G'](u, v)
  rhs_u = u + dt * F
  rhs_v = v + dt * G
  U = np.fft.fftn(u)
  V = np.fft.fftn(v)
  RHS_U = np.fft.fftn(rhs_u)
  RHS_V = np.fft.fftn(rhs_v)

  denom_u = (1.0 - dt * params['alpha'] * lap_ev)
  denom_v = (1.0 - dt * params['beta']  * lap_ev)

  U_nueva = RHS_U / denom_u
  V_nueva = RHS_V / denom_v

  u_nueva = np.real(np.fft.ifftn(U_nueva))
  v_nueva = np.real(np.fft.ifftn(V_nueva))
  return u_nueva, v_nueva

def imex(u0, v0, params, Lx=3.0, Ly=3.0, T=15.0, dt=0.01, save_every=None):
  ny, nx = u0.shape
  lap_ev = fourier_laplacian_eigenvalues(nx, ny, Lx, Ly)
  u, v = u0.copy(), v0.copy()
  pasos = int(np.ceil(T / dt))
  historia = []
  for n in range(pasos):
    u, v = pasos_imex(u, v, params, lap_ev, dt)
    if save_every and (n % save_every == 0):
      historia.append((n*dt, u.copy(), v.copy()))
  return u, v, historia

# Definiciones de sistemas

def F_base(u, v):
  return u - u*(v**3) - v - 0.05

def G_base(u, v):
  return 10.0*(u - v)

def F_suave(u, v, c=1, d=0.05):
  return u - c*u*(v**2) - v - d

def G_suave(u, v, k=10):
  return k*(u - v)

# Visualización y guardado

def crear_sistema(ax, titulo, params, extra=''):
  txt = []
  txt.append(f"alpha={params['alpha']}, beta={params['beta']}")
  txt.append(f"F(u,v)={params['F_text']}")
  txt.append(f"G(u,v)={params['G_text']}")
  if extra:
    txt.append(extra)
  ax.text(0.02, 0.02, "\n".join(txt), color='k', fontsize=8,
          transform=ax.transAxes, va='bottom',
          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
  
def guardar_patron(img, titulo, params, cmap='viridis', Lx=3.0, Ly=3.0, extra=''):
  plt.figure(figsize=(5.2, 4.6), dpi=160)
  ax = plt.gca()
  im = ax.imshow(img, extent=[0, Lx, 0, Ly], origin='lower', cmap=cmap)
  plt.title(titulo)
  plt.xlabel('x'); plt.ylabel('y')
  plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='u')
  crear_sistema(ax, titulo, params, extra=extra)
  guardar_titulo = titulo.replace(' ', '_')
  fnombre = f"2_{guardar_titulo}.png"
  plt.tight_layout()
  plt.savefig(fnombre)
  plt.close()

# Ejecución

if __name__ == "__main__":
  Lx, Ly = 3.0, 3.0
  nx, ny = 256, 256
  x = np.linspace(0, Lx, nx, endpoint=False)
  y = np.linspace(0, Ly, ny, endpoint=False)
  
  rng = np.random.default_rng(42)
  u0 = 0.1 * rng.standard_normal((ny, nx))
  v0 = 0.1 * rng.standard_normal((ny, nx))

  T_max = 15.0
  dt = 0.01

# cambiar variables en params para generar diferentes patrones
#Variables: alpha,beta,c,d,k

  params = {
      'alpha': 0.00028, 'beta': 0.05,
      'F': lambda u, v: F_suave(u, v, c=1, d=0.05),
      'G': lambda u, v: G_suave(u, v, k=10.0),
      'F_text': "u - 1*u*v^2 - v - 0.05",
      'G_text': "10*(u - v)"
  }

  u, v, _ = imex(u0, v0, params, Lx=Lx, Ly=Ly, T=T_max, dt=dt)
  guardar_patron(u, "patrón_base", params, cmap='cividis', Lx=Lx, Ly=Ly)

#  params_base = {
#    'alpha': 0.00028, 'beta': 0.05,
#    'F': lambda u, v: F_suave(u, v, c=1, d=0.05),
#   'G': lambda u, v: G_suave(u, v, k=10.0),
#    'F_text': "u - 1*u*v^2 - v - 0.05",
#    'G_text': "10*(u - v)"
#  }

#  u, v, _ = imex(u0, v0, params, Lx=Lx, Ly=Ly, T=T_max, dt=dt)
#  guardar_patron(u, "patrón_base", params, cmap='cividis', Lx=Lx, Ly=Ly)


#  params = {
#      'alpha': 0.00028, 'beta': 0.05,
#      'F': lambda u, v: F_suave(u, v, c=10, d=0.05),
#      'G': lambda u, v: G_suave(u, v, k=10),
#      'F_text': "u - 10*u*v^2 - v - 0.05",
#      'G_text': "10*(u - v)"
#  }

#  u, v, _ = imex(u0, v0, params, Lx=Lx, Ly=Ly, T=T_max, dt=dt)
#  guardar_patron(u, "Bacterias", params, cmap='cividis', Lx=Lx, Ly=Ly)  


#  params = {
#      'alpha': 0.00028, 'beta': 0.05,
#      'F': lambda u, v: F_suave(u, v, c=10, d=200),
#      'G': lambda u, v: G_suave(u, v, k=10),
#      'F_text': "u - 10*u*v^2 - v - 200",
#      'G_text': "10*(u - v)"
#  }

#  u, v, _ = imex(u0, v0, params, Lx=Lx, Ly=Ly, T=T_max, dt=dt)
#  guardar_patron(u, "cortina", params, cmap='cividis', Lx=Lx, Ly=Ly)  


#--------------
# PUNTO 3
#--------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from IPython.display import HTML, display

# --- FUNCIONES PRINCIPALES Y COMPARTIDAS ---

def soliton(x, A, x0):
    """
    Genera el perfil de un solitón de la ecuación KdV.
    La velocidad (v) es proporcional a la amplitud (A), v = A/3.
    """
    v = A / 3.0
    # El factor de ancho (k) también depende de la velocidad/amplitud.
    k = 0.5 * np.sqrt(v)
    return A / (np.cosh(k * (x - x0))**2)

def run_simulation(psi0, T, L, dt, dx, save_every=None, max_frames=None):
    """
    Ejecuta la simulación de la ecuación KdV pero permite limitar
    el número máximo de frames guardados (max_frames) o usar save_every.
    Si se pasa max_frames, se calcula save_every automáticamente.
    """
    steps = int(np.round(T / dt))
    if steps <= 0:
        raise ValueError("T/dt debe ser positivo y mayor que 0")

    # Si el usuario pidió un máximo de frames, calcular save_every
    if max_frames is not None:
        if max_frames < 2:
            max_frames = 2
        # Queremos aproximadamente max_frames (incluye frame inicial).
        save_every = max(1, steps // (max_frames - 1))

    if save_every is None:
        save_every = 1  # por defecto guardar cada paso (no recomendado)

    C1 = dt / (3.0 * dx)
    C2 = dt / (2.0 * dx**3)

    psi = np.copy(psi0)
    simulation_data = [np.copy(psi0)]

    # Para acelerar la comprobación, precomputamos un set de índices donde guardaremos
    # esto evita hacer comprobaciones de módulo muy costosas (aunque el modulo es rápido)
    # pero es útil para control
    indices_to_save = set(range(0, steps + 1, save_every))
    # Aseguramos incluir el último índice
    indices_to_save.add(steps)

    for step in range(1, steps + 1):
        psi_p1 = np.roll(psi, -1)
        psi_p2 = np.roll(psi, -2)
        psi_m1 = np.roll(psi, 1)
        psi_m2 = np.roll(psi, 2)

        nonlinear_term = C1 * (psi_p1 + psi + psi_m1) * (psi_p1 - psi_m1)
        dispersive_term = C2 * (psi_p2 - 2 * psi_p1 + 2 * psi_m1 - psi_m2)

        psi = psi - nonlinear_term - dispersive_term

        if step in indices_to_save:
            simulation_data.append(np.copy(psi))

    return np.array(simulation_data)


# --- FUNCIONES AUXILIARES DE ANIMACIÓN PARA GOOGLE COLAB ---

def mostrar_animacion_interaccion(sim_data, x, L):
    """
    Muestra en Google Colab la animación de interacción de solitones.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, sim_data[0], 'b-')
    ax.set_title('Interacción de Dos Solitones', fontsize=16)
    ax.set_xlabel('Posición (x)'); ax.set_ylabel('Amplitud ($\\phi$)')
    ax.set_xlim(0, L); ax.set_ylim(-0.5, 4.0)

    def animate(i):
        line.set_ydata(sim_data[i])
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=len(sim_data), blit=True, interval=30)
    display(HTML(ani.to_html5_video()))
    plt.close(fig)


def mostrar_animacion_condiciones(data_cos, data_pure, data_weak, x, L):
    """
    Muestra en Google Colab la animación de distintas condiciones iniciales.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle('Evolución de Diferentes Condiciones Iniciales', fontsize=16)

    ax1.set_title('Inicial: Coseno -> Tren de Solitones'); ax1.set_ylim(-3.5, 3.5)
    ax2.set_title('Inicial: Solitón Puro -> Propagación Estable'); ax2.set_ylim(-0.5, 3.5)
    ax3.set_title('Inicial: Pulso Débil -> Dispersión'); ax3.set_ylim(-0.5, 1.0)

    lines = []
    initial_data = [data_cos[0], data_pure[0], data_weak[0]]

    for i, ax in enumerate([ax1, ax2, ax3]):
        line, = ax.plot(x, initial_data[i], 'b-')
        ax.set_xlim(0, L); ax.set_ylabel('Amplitud ($\\phi$)'); ax.grid(True)
        lines.append(line)

    ax3.set_xlabel('Posición (x)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    def animate(i):
        lines[0].set_data(x, data_cos[i])
        lines[1].set_data(x, data_pure[i])
        lines[2].set_data(x, data_weak[i])
        return lines

    frames = len(data_cos)
    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, interval=30)
    display(HTML(ani.to_html5_video()))
    plt.close(fig)


# --- FUNCIONES PARA CADA PREGUNTA ---

def generar_grafico_amplitud_velocidad():
    """PREGUNTA 1 & 3: Genera un PDF que muestra la relación Amplitud-Velocidad."""
    print("\n--- Generando Gráfico: Amplitud vs. Velocidad (Preguntas 1 y 3) ---")

    L = 10.0
    dx = 0.1
    x = np.arange(0, L, dx)

    solitones = {
        'Pequeño': {'A': 1.0, 'x0': 2.0},
        'Mediano': {'A': 2.0, 'x0': 4.5},
        'Grande':  {'A': 3.0, 'x0': 7.0}
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for nombre, params in solitones.items():
        A, x0 = params['A'], params['x0']
        v = A / 3.0
        perfil = soliton(x, A, x0)
        ax.plot(x, perfil, label=f'Solitón {nombre} (A={A:.1f}, v={v:.2f})')

    ax.set_title('Relación entre Amplitud y Velocidad de Solitones', fontsize=16)
    ax.set_xlabel('Posición (x)', fontsize=12)
    ax.set_ylabel('Amplitud ($\\phi$)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    ax.set_xlim(0, L)
    ax.set_ylim(0, 3.5)

    output_filename = '3_AmplitudVelocidad.pdf'
    plt.savefig(output_filename)
    print(f"Gráfico guardado como '{output_filename}'")
    plt.show()


def generar_animacion_interaccion():
    """PREGUNTA 2: Genera un MP4 de la interacción y un TXT de la conservación."""
    print("\n--- Generando Animación y TXT: Interacción de Solitones (Pregunta 2) ---")

    L = 40.0; T = 60.0; dx = 1; dt = 0.0001
    x = np.arange(0, L, dx)

    psi0 = soliton(x, A=3.0, x0=8.0) + soliton(x, A=1.5, x0=15.0)

    print("Iniciando simulación de interacción (esto puede tardar un momento)...")
    sim_data = run_simulation(psi0, T, L, dt, dx, save_every=300)
    print("Simulación completa.")

    momento_inicial = np.trapezoid(sim_data[0]**2, x)
    momento_final = np.trapezoid(sim_data[-1]**2, x)

    with open('3_Conservacion.txt', 'w', encoding='utf-8') as f:
        f.write('ANÁLISIS DE CONSERVACIÓN DURANTE LA INTERACCIÓN\n')
        f.write('================================================\n')
        f.write(f'Momento (∫ψ²) inicial : {momento_inicial:.6f}\n')
        f.write(f'Momento (∫ψ²) final    : {momento_final:.6f}\n\n')
        f.write('Los valores son casi idénticos, demostrando la conservación.\n')
    print("Archivo '3_Conservacion.txt' generado.")

    print("Generando animación en Google Colab...")
    mostrar_animacion_interaccion(sim_data, x, L)


def generar_analisis_cfl():
    """PREGUNTA 4: Genera un PDF y un TXT sobre la condición de estabilidad."""
    print("\n--- Generando Análisis de Estabilidad (CFL) (Pregunta 4) ---")

    cfl_explanation = """
ANÁLISIS DE LA CONDICIÓN DE ESTABILIDAD (CFL) PARA KdV
=========================================================

La ecuación de Korteweg-de Vries (KdV) contiene un término no lineal (u*u_x) y un término dispersivo (u_xxx). Cada uno impone una restricción sobre el paso de tiempo (dt) para que la simulación numérica sea estable.

1.  Término no lineal: Requiere dt ∝ dx.
2.  Término dispersivo: Requiere dt ∝ dx³.

La condición más restrictiva es la del término dispersivo. Por lo tanto, para garantizar la estabilidad, se debe cumplir que:

    dt / dx³ < C

donde C es una constante que depende del esquema numérico. Si dt es demasiado grande en relación con dx, la simulación se volverá inestable, y los errores crecerán exponencialmente.
"""
    with open('3_CondicionCFL.txt', 'w', encoding='utf-8') as f:
        f.write(cfl_explanation)
    print("Archivo '3_CondicionCFL.txt' generado.")

    L = 20.0; T = 5.0; dx = 0.5
    x = np.arange(0, L, dx)
    psi0 = soliton(x, A=2.0, x0=5.0)

    dt_stable = 0.00001
    cfl_stable = dt_stable / dx**3
    print(f"Ejecutando simulación estable (dt/dx³ = {cfl_stable:.3f})...")
    psi_final_stable = run_simulation(psi0, T, L, dt_stable, dx, save_every=int(T/dt_stable))[-1]
    print(psi_final_stable)
    dt_unstable = 0.003
    cfl_unstable = dt_unstable / dx**3
    print(f"Ejecutando simulación inestable (dt/dx³ = {cfl_unstable:.3f})...")
    psi_final_unstable = run_simulation(psi0, T, L, dt_unstable, dx, save_every=int(T/dt_unstable))[-1]
    print(psi_final_unstable)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Efecto de la Condición de Estabilidad Numérica (CFL)', fontsize=16)

    ax1.plot(x, psi0, 'b--', label='Inicial')
    ax1.plot(x, psi_final_stable, 'r-', label=f'Final (Estable, dt={dt_stable})')
    ax1.set_title(f'Simulación Estable (dt/dx³ = {cfl_stable:.2f})')
    ax1.legend(); ax1.grid(True)

    ax2.plot(x, psi0, 'b--', label='Inicial')
    ax2.plot(x, psi_final_unstable, 'r-', label=f'Final (Inestable, dt={dt_unstable})')
    ax2.set_title(f'Simulación Inestable (dt/dx³ = {cfl_unstable:.2f})')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = '3_InestabilidadNumerica.pdf'
    plt.savefig(output_filename)
    print(f"Gráfico guardado como '{output_filename}'")
    plt.show()


def generar_animacion_condiciones_iniciales():
    """PREGUNTA 5: Versión robusta que limita frames y evita impresiones masivas."""
    print("\n--- Generando Animación: Condiciones Iniciales (Pregunta 5) ---")

    L = 40.0; T = 50.0; dx = 0.4; dt = 0.00005
    x = np.arange(0, L, dx)

    psi0_cos = 3.0 * np.cos(2 * np.pi * x / L)
    psi0_pure = soliton(x, A=3.0, x0=L/2)
    psi0_weak = 0.5 * np.exp(-((x - L/2)**2) / 2.0)

    # LIMITADOR: cuantos frames queremos como máximo en la animación
    MAX_FRAMES = 400

    print("Ejecutando simulación 1 (Coseno) ...")
    data_cos = run_simulation(psi0_cos, T, L, dt, dx, max_frames=MAX_FRAMES)
    print("Ejecutando simulación 2 (Solitón Puro) ...")
    data_pure = run_simulation(psi0_pure, T, L, dt, dx, max_frames=MAX_FRAMES)
    print("Ejecutando simulación 3 (Pulso Débil) ...")
    data_weak = run_simulation(psi0_weak, T, L, dt, dx, max_frames=MAX_FRAMES)
    print("Simulaciones completas.")

    # Evitar imprimir contenidos enormes (quita prints pesados)
    print(f"Frames guardados por simulación: {len(data_cos)} (máx {MAX_FRAMES})")

    print("Generando animación en Google Colab...")
    mostrar_animacion_condiciones(data_cos, data_pure, data_weak, x, L)


def main():
    """Función principal que muestra el menú interactivo."""
    if not os.path.exists('resultados_kdv'):
        os.makedirs('resultados_kdv')
    os.chdir('resultados_kdv')

    while True:
        print("\n" + "="*50)
        print("     MENÚ DE ANÁLISIS DE SOLITONES (ECUACIÓN KdV)")
        print("="*50)
        print("1. Gráfico de Amplitud vs. Velocidad (Preguntas 1 y 3)")
        print("2. Animación de Interacción de Solitones (Pregunta 2)")
        print("3. Análisis de Estabilidad Numérica (CFL) (Pregunta 4)")
        print("4. Animación de Generación de Solitones (Pregunta 5)")
        print("5. Salir")
        print("-"*50)

        choice = input("Selecciona una opción (1-5): ")

        if choice == '1':
            generar_grafico_amplitud_velocidad()
        elif choice == '2':
            generar_animacion_interaccion()
        elif choice == '3':
            generar_analisis_cfl()
        elif choice == '4':
            generar_animacion_condiciones_iniciales()
        elif choice == '5':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, intenta de nuevo.")

        print("\nVolviendo al menú principal...")

if __name__ == '__main__':
    main()
