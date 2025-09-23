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