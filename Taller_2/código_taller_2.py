import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.signal import peak_widths
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.stats import linregress
from numpy.typing import NDArray

#=========
# PUNTO 1
#=========

#1a
@njit
def Fourier_transform(t: NDArray[np.float64],
                      y: NDArray[np.float64],
                      f: NDArray[np.float64]) -> NDArray[np.complex128]:
    """
    Calcula la transformada discreta de Fourier en frecuencias arbitrarias f.
    Definición: F_k = sum_{i=1}^N y_i * exp(-2π i f_k t_i)
    """
    N = len(t)
    resultados = np.zeros(len(f), dtype=np.complex128)
    for k in range(len(f)):
        exp_term = np.exp(-2j * np.pi * f[k] * t)
        resultados[k] = np.sum(y * exp_term)
    return resultados


# === Generación de señales de prueba ===
t_max = 1.0
dt = 0.001
t = np.arange(0, t_max, dt)

# Señal con varias frecuencias
amplitudes = np.array([1.0, 0.5, 0.3])
frecuencias_componentes = np.array([5.0, 15.0, 30.0])

y = np.zeros_like(t)
for A, f in zip(amplitudes, frecuencias_componentes):
    y += A * np.sin(2 * np.pi * f * t)

# Señal con ruido
ruido = 0.5
y_ruido = y + ruido * np.random.randn(len(t))

# Definir las frecuencias a evaluar (no tienen que coincidir con el tamaño de los datos)
f_eval = np.linspace(0, 50, 500)

# Calcular la transformada en esas frecuencias
F = Fourier_transform(t, y, f_eval)
F_ruido = Fourier_transform(t, y_ruido, f_eval)

# === Graficar ===
plt.figure(figsize=(12, 5))

# Señal sin ruido
plt.subplot(1, 2, 1)
plt.plot(f_eval, np.abs(F), label="Magnitud |F|")
plt.title("Espectro sin ruido")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.legend()

# Señal con ruido
plt.subplot(1, 2, 2)
plt.plot(f_eval, np.abs(F_ruido), label="Magnitud |F| con ruido")
plt.title("Espectro con ruido")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("1.a.pdf")
#plt.show()


#1b
def generar_senal(t, freq, SNtime):
    A = SNtime * freq
    signal = A * np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, 1, len(t))
    return signal + noise, signal, noise

def calcular_SNfreq(t, y, freq, f_eval):
    Y = Fourier(t, y, f_eval)
    P = np.abs(Y)
    peak = np.max(P)
    mask = (f_eval < freq*0.8) | (f_eval > freq*1.2)
    noise_std = np.std(P[mask])
    return peak / noise_std

# Parámetros
tmax = 5
dt = 0.01
t = np.arange(0, tmax, dt)
freq = 2.0
f_eval = np.linspace(0, 10, 2000)

SNtime_values = np.logspace(-2, 0, 20)
SNfreq_values = []

for SNtime in SNtime_values:
    y, s, n = generar_senal(t, freq, SNtime)
    SNfreq = calcular_SNfreq(t, y, freq, f_eval)
    SNfreq_values.append(SNfreq)

SNfreq_values = np.array(SNfreq_values)

# Ajuste en log-log
log_SNtime = np.log10(SNtime_values)
log_SNfreq = np.log10(SNfreq_values)
slope, intercept, r_value, _, _ = linregress(log_SNtime, log_SNfreq)
fit_line = 10**(intercept + slope * log_SNtime)

plt.figure(figsize=(7,5))
plt.loglog(SNtime_values, SNfreq_values, 'o-', label="Datos simulados")
plt.loglog(SNtime_values, fit_line, '--', label=f"Ajuste ~ SNtime^{slope:.2f}")
plt.xlabel("SNtime")
plt.ylabel("SNfreq")
plt.title("Relación SNfreq vs SNtime")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.savefig("1.b.pdf")
#plt.show()

print(f"Modelo encontrado: SNfreq ≈ (SNtime^{slope:.2f}) * {10**intercept:.2f}")

#1c
def generar_datos(A=1.0, freq=5.0, tmax=10.0, dt=0.01, SNtime=0.5):
    t = np.arange(0, tmax, dt)
    señal = A * np.sin(2 * np.pi * freq * t)
    ruido = (A / SNtime) * np.random.normal(0, 1, len(t))
    return t, señal + ruido

def medir_fwhm(t, y, freq):
    if len(t) < 2:   # chequeo para evitar error
        return None
    N = len(t)
    dt = t[1] - t[0]
    freqs = fftfreq(N, dt)
    Y = np.abs(fft(y))**2
    mask = freqs > 0
    freqs = freqs[mask]
    Y = Y[mask]
    idx_peak = np.argmax(Y)
    peak_height = Y[idx_peak]
    half_max = peak_height / 2
    left_idx = np.where(Y[:idx_peak] < half_max)[0]
    right_idx = np.where(Y[idx_peak:] < half_max)[0]
    if len(left_idx) == 0 or len(right_idx) == 0:
        return None
    left = freqs[left_idx[-1]]
    right = freqs[idx_peak + right_idx[0]]
    return right - left

A = 1.0
freq = 5.0
dt = 0.01
SNtime = 0.5
tmax_values = np.linspace(-30, 15, 30)  # ⬅ empieza en 0.5 en lugar de 0
fwhm_values = []

for tmax in tmax_values:
    t, y = generar_datos(A=A, freq=freq, tmax=tmax, dt=dt, SNtime=SNtime)
    fwhm = medir_fwhm(t, y, freq)
    fwhm_values.append(fwhm if fwhm is not None else np.nan)

plt.figure(figsize=(7,5))
plt.plot(tmax_values, fwhm_values, 'o-', label="Ancho del pico (FWHM)")
plt.xlabel("tmax (s)")
plt.ylabel("Ancho del pico (Hz)")
plt.title("Dependencia del ancho del pico en función de tmax")
plt.grid(True)
plt.legend()
plt.savefig("1.c.pdf")
#plt.show()

#Bono
def plot_aliasing(f_signal=50, duracion=0.1):
    fs_values = [200, 120, 80, 60]
    t_cont = np.linspace(0, duracion, 5000)
    y_cont = np.sin(2*np.pi*f_signal*t_cont)
    fig, axs = plt.subplots(len(fs_values), 1, figsize=(8, 8), sharex=True)
    for i, fs in enumerate(fs_values):
        t = np.arange(0, duracion, 1/fs)
        y = np.sin(2*np.pi*f_signal*t)
        axs[i].plot(t_cont, y_cont, 'k--', label='Señal continua')
        axs[i].stem(t, y, linefmt='C0-', markerfmt='C0o', basefmt=" ", label=f'Muestreo fs={fs} Hz')
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_ylabel("Amplitud")
    axs[-1].set_xlabel("Tiempo (s)")
    plt.suptitle("BONO: Aliasing al muestrear más allá de Nyquist", fontsize=14)
    plt.tight_layout()
    plt.savefig("BONO.pdf")
    #plt.show()

plot_aliasing()

#=========
# PUNTO 2
#=========

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# 2.a. Arreglar datos

# Cargar el archivo, ignorando comentarios (#) y usando el delimitador correcto (;)

df = pd.read_csv("/SN_d_tot_V2.0.csv", sep=",", comment="#")

# Columnas según SILSO (seleccionar las primeras 5 columnas)

df = df.iloc[:, :5]
df.columns = ["year", "month", "day", "decimal_date", "sunspots"]

# Reemplazar -1 por NaN y luego interpolar linealmente

df["sunspots"] = df["sunspots"].replace(-1, np.nan)
df["sunspots"] = df["sunspots"].interpolate(method="linear")

# 2.b. FFT y periodo

y = df["sunspots"].values
N = len(y)

# Serie de tiempo: un punto por día

t = np.arange(N)

# FFT (centrada en la media)

Y = fft(y - np.mean(y))
freqs = fftfreq(N, d=1)   # en ciclos por día

# Solo frecuencias positivas
mask = freqs > 0
freqs = freqs[mask]
power = np.abs(Y[mask])**2

# Limitar a ciclos entre 8 y 16 años (≈3000 a 6000 días)
min_freq = 1/6000
max_freq = 1/3000
relevant_freqs_mask = (freqs >= min_freq) & (freqs <= max_freq)

# Encontrar el máximo en ese rango
peak_idx = np.argmax(power[relevant_freqs_mask])
original_indices_of_relevant_freqs = np.where(relevant_freqs_mask)[0]
peak_freq_original_idx = original_indices_of_relevant_freqs[peak_idx]

f_peak = freqs[peak_freq_original_idx]
period_days = 1 / f_peak

# Guardar en archivo de texto
with open("2.b.txt", "w") as f:
    f.write(f"{period_days:.2f} días (~{period_days/365:.2f} años)\n")

# Filtrado pasa bajas
filtered = gaussian_filter1d(y, sigma=500)

plt.figure(figsize=(12,6))
plt.plot(df["decimal_date"], y, lw=0.3, label="Datos originales")
plt.plot(df["decimal_date"], filtered, lw=2, label="Filtrado (Gauss)")
plt.xlabel("Fecha decimal (años)")
plt.ylabel("Número de manchas solares")
plt.legend()
plt.tight_layout()
plt.savefig("2.b.data.pdf")
plt.close()

# Máximos locales
# distancia mínima entre máximos ~3000 días (~8 años)
peaks, _ = find_peaks(filtered, distance=3000)
peak_times = df["decimal_date"].iloc[peaks].values
peak_values = filtered[peaks]

plt.figure(figsize=(10,6))
plt.scatter(peak_times, peak_values, color="red", label="Máximos locales")
plt.plot(df["decimal_date"], filtered, color="blue", lw=1, label="Filtrado")
plt.xlabel("Fecha decimal (años)")
plt.ylabel("Número de manchas solares (filtrado)")
plt.title("Máximos locales del ciclo solar")
plt.legend()
plt.tight_layout()
plt.savefig("2.b.maxima.pdf")
plt.close(

#=========
# PUNTO 3
#=========

#3.a.
def desenfoque_gaussiano(imagen_path, A):
  imagen = np.array(Image.open(imagen_path).convert("RGB"), dtype=float)
  imagen_borrosa = np.zeros_like(imagen)

  for c in range(3):
    canal = imagen[:, :, c]
    # FFT centrada
    F = fft2(canal)
    F_shift = fftshift(F)

    # Filtro gaussiano
    h, w = canal.shape
    y, x = np.indices((h, w))
    cx, cy = w // 2, h // 2
    gauss = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * A**2))

    # Filtrado en frecuencia
    F_filtrado = F_shift * gauss

    # Transformada inversa
    imagen_borrosa[:, :, c] = np.abs(ifft2(ifftshift(F_filtrado)))

  Image.fromarray(np.uint8(imagen_borrosa)).save("3.a.jpg")

desenfoque_gaussiano("miette.jpg", 20)

#3.b.a.

def eliminar_ruido_periodico(ruta_imagen_entrada, percentil=99.9, radio=3):
  imagen = Image.open(ruta_imagen_entrada).convert('L')
  imagen_array = np.array(imagen, dtype=float)

  F = fft2(imagen_array)
  F_shift = fftshift(F)
  magnitud = np.abs(F_shift)
  mag_log = np.log1p(magnitud)

  filas, columnas = imagen_array.shape
  centro = (filas // 2, columnas // 2)
  umbral = np.percentile(mag_log, percentil)
  coords = peak_local_max(
      mag_log,
      min_distance=10,
      threshold_abs=umbral,
      exclude_border=False
  )

  coords_filtradas = [
      (int(y), int(x)) for y, x in coords
      if (abs(y - centro[0]) > 5 or abs(x - centro[1]) > 5)
  ]

  for (y, x) in coords_filtradas:
    for dy in range(-radio, radio + 1):
      for dx in range(-radio, radio + 1):
        yy = (y + dy) % filas
        xx = (x + dx) % columnas
        sy = (2 * centro[0] - yy) % filas
        sx = (2 * centro[1] - xx) % columnas
        F_shift[yy, xx] = 0
        F_shift[sy, sx] = 0


  F_ishift = ifftshift(F_shift)
  imagen_sin_ruido = np.abs(ifft2(F_ishift))

  imagen_norm = 255 * (imagen_sin_ruido - imagen_sin_ruido.min()) / np.ptp(imagen_sin_ruido)
  imagen_norm = np.uint8(imagen_norm)


  Image.fromarray(imagen_norm).save("3.b.a.jpg")

eliminar_ruido_periodico("p_a_t_o.jpg", percentil=99.9, radio=5)

#3.b.b.

def detectar_picos(fft_magnitud, umbral):
  h, w = fft_magnitud.shape
  cy, cx = h // 2, w // 2
  coords = []
  for y in range(h):
    for x in range(w):
      if abs(y - cy) < 5 and abs(x - cx) < 5:
        continue  # no tocar la componente DC
      if fft_magnitud[y, x] > umbral:
        coords.append((y, x))
  return coords

def eliminar_ruido_gato(imagen_path, factor_umbral):
  imagen = np.array(Image.open(imagen_path).convert("L"), dtype=float)
  F = fft2(imagen)
  F_shift = fftshift(F)
  magnitud_log = np.log1p(np.abs(F_shift))

  umbral = magnitud_log.mean() * factor_umbral
  picos = detectar_picos(magnitud_log, umbral)

  h, w = imagen.shape
  cy, cx = h // 2, w // 2
  for (y, x) in picos:
    F_shift[y, x] = 0
    sy = (2*cy - y) % h
    sx = (2*cx - x) % w
    F_shift[sy, sx] = 0

  imagen_sin_ruido = np.abs(ifft2(ifftshift(F_shift)))
  imagen_sin_ruido = np.clip(imagen_sin_ruido, 0, 255)
  Image.fromarray(np.uint8(imagen_sin_ruido)).save("3.b.b.png")

eliminar_ruido_gato("g_a_t_o.png", 5)

#=========
# PUNTO 4
#=========

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.timeseries import LombScargle

ruta = Path(r"C:\Users\usuario\OneDrive\Escritorio\Universidad\Materias\Metodos Computacionales 2\mammography_spectra\OGLE-LMC-CEP-0001.dat")

if not Path(ruta).exists():
    raise FileNotFoundError(f"No encuentro el archivo en: {ruta}")

# === 2) Leer datos: tiempo (t), magnitud (m), incertidumbre (dm)
t, m, dm = np.loadtxt(ruta, unpack=True, comments="#")

# === 3) Periodograma Lomb-Scargle (datos irregulares)
# Rango razonable para una cefeida: 0.01–2 ciclos/día (ajustable)
min_freq = 0.01
max_freq = 2.0
freqs = np.linspace(min_freq, max_freq, 10000)

ls = LombScargle(t, m, dm)
power = ls.power(freqs)

best_freq = freqs[np.argmax(power)]
best_period = 1.0 / best_freq

print(f"Frecuencia dominante: {best_freq:.6f} ciclos/día")
print(f"Período correspondiente: {best_period:.6f} días")

# === 4) Calcular fase y graficar brillo vs fase
phi = np.mod(best_freq * t, 1.0)

plt.figure(figsize=(7,5))
plt.errorbar(phi, m, yerr=dm, fmt='.', ms=4, alpha=0.8, label="Datos")
plt.gca().invert_yaxis()  # magnitud: menor = más brillante
plt.xlabel("Fase (ϕ)")
plt.ylabel("Magnitud")
plt.title(f"Brillo vs fase  |  f = {best_freq:.6f} c/d,  P = {best_period:.3f} d")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("4.pdf")
#plt.show()

#=========
# PUNTO 5
#=========

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from numpy.fft import rfft, irfft, rfftfreq

# --- Paths ---
data_path = "/mnt/data/tomography_data/tomography_data/3.npy"
out_unfiltered = "/mnt/data/4_unfiltered.png"
out_filtered = "/mnt/data/4.png"

# --- Load projections ---
projections = np.load(data_path)  # shape (n_angles, n_detectors) ?
print("Shape:", projections.shape)

# Si está al revés, transponer
if projections.shape[0] < projections.shape[1]:
    projections = projections.T
    print("Transposed:", projections.shape)

n_angles, n_detectors = projections.shape
rows = n_detectors
angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)

# --- High-pass filter (ramp + Hann) ---
freqs = rfftfreq(n_detectors, d=1.0)
ramp = np.abs(freqs)
hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(len(ramp)) / (len(ramp) - 1))
ramp_windowed = ramp * hann

filtered = np.empty_like(projections, dtype=float)
for i in range(n_angles):
    p = projections[i].astype(float)
    P = rfft(p)
    Pf = P * ramp_windowed
    filtered[i] = irfft(Pf, n=n_detectors)

# --- Reconstrucción SIN filtro ---
accum_unfiltered = np.zeros((rows, rows))
for angle, signal in zip(angles, projections):
    imagen_rotada = ndi.rotate(
        np.tile(signal[:, None], rows).T,
        angle,
        reshape=False,
        mode="reflect"
    )
    accum_unfiltered += imagen_rotada

# --- Reconstrucción CON filtro ---
accum_filtered = np.zeros((rows, rows))
for angle, signal in zip(angles, filtered):
    imagen_rotada = ndi.rotate(
        np.tile(signal[:, None], rows).T,
        angle,
        reshape=False,
        mode="reflect"
    )
    accum_filtered += imagen_rotada

# --- Normalizar ---
def normalize(img):
    img = img - img.min()
    return img / img.max() if img.max() > 0 else img

accum_unfiltered = normalize(accum_unfiltered)
accum_filtered = normalize(accum_filtered)

# --- Guardar imágenes ---
plt.imsave(out_unfiltered, accum_unfiltered, cmap="gray", origin="lower")
plt.imsave(out_filtered, accum_filtered, cmap="gray", origin="lower")

print("Reconstrucciones guardadas en:")
print(" -", out_unfiltered)
print(" -", out_filtered)
