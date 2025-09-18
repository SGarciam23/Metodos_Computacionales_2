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

def F_bias(u, v, bias=-0.03):
  return u - u*(v**3) - v + bias

def G_bias(u, v, k=12.0):
  return k*(u - v)

def F_suave(u, v, c=0.9, d=0.04):
  return u - c*u*(v**2) - v - d

def G_suave(u, v, k=8.0):
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
