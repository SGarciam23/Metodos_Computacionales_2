
#-----------------
#PUNTO 2
#-----------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Ruta base
ruta_base = r"C:\Users\usuario\OneDrive\Escritorio\Universidad\Materias\Metodos Computacionales 2\mammography_spectra"

# Listas para guardar ejemplos
ejemplos_remocion = []
ejemplos_gaussiana = []
espectros_aproximados = []

# Número de puntos vecinos a eliminar alrededor del pico
puntos_vecinos = 3

# Función gaussiana
def gaussiana(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0)*2 / (2 * sigma*2)) + c

# Recorrer subcarpetas y archivos
for subcarpeta in os.listdir(ruta_base):
    ruta_subcarpeta = os.path.join(ruta_base, subcarpeta)
    if not os.path.isdir(ruta_subcarpeta):
        continue

    for archivo in os.listdir(ruta_subcarpeta):
        if not archivo.endswith(".dat"):
            continue

        # Leer datos
        try:
            df = pd.read_csv(os.path.join(ruta_subcarpeta, archivo),
                             delim_whitespace=True, encoding='latin1', header=None, comment='#')
        except Exception as e:
            print(f"No se pudo leer {archivo}: {e}")
            continue

        energia = df.iloc[:, 0].values
        conteo = df.iloc[:, 1].values

        # Detectar picos y quitar vecinos
        picos, _ = find_peaks(conteo, prominence=0.05 * np.max(conteo))
        mascara = np.ones_like(conteo, dtype=bool)
        for p in picos:
            ini = max(0, p - puntos_vecinos)
            fin = min(len(conteo), p + puntos_vecinos + 1)
            mascara[ini:fin] = False

        energia_sin_picos = energia[mascara]
        conteo_sin_picos = conteo[mascara]

        # Guardar ejemplos de limpieza
        if len(ejemplos_remocion) < 3:
            ejemplos_remocion.append((energia, conteo, energia_sin_picos, conteo_sin_picos, archivo))

        # Ajuste gaussiano
        try:
            p0 = [np.max(conteo_sin_picos), energia_sin_picos[np.argmax(conteo_sin_picos)],
                  (energia_sin_picos[-1]-energia_sin_picos[0])/6, np.min(conteo_sin_picos)]
            popt, _ = curve_fit(gaussiana, energia_sin_picos, conteo_sin_picos, p0=p0)
            conteo_gauss = gaussiana(energia, *popt)
        except Exception as e:
            print(f"No se pudo ajustar {archivo}: {e}")
            conteo_gauss = conteo_sin_picos

        # Guardar ejemplos de gaussiana
        if len(ejemplos_gaussiana) < 3:
            ejemplos_gaussiana.append((energia, conteo_gauss, archivo))

        # Transformación para aproximar curvas entre sí
        # Normalizar entre 0 y 1
        conteo_norm = (conteo_gauss - np.min(conteo_gauss)) / (np.max(conteo_gauss) - np.min(conteo_gauss))
        # Centrar energía en el máximo para aproximar curvas
        energia_centrada = energia - energia[np.argmax(conteo_gauss)]

        espectros_aproximados.append((energia_centrada, conteo_norm, archivo))

# -------------------- Seleccionar 1 curva representativa por elemento --------------------
# Suponiendo que cada subcarpeta es un elemento
representativos = {}
for subcarpeta in os.listdir(ruta_base):
    ruta_subcarpeta = os.path.join(ruta_base, subcarpeta)
    if not os.path.isdir(ruta_subcarpeta):
        continue

    for archivo in os.listdir(ruta_subcarpeta):
        if not archivo.endswith(".dat"):
            continue

        # Leer datos
        try:
            df = pd.read_csv(os.path.join(ruta_subcarpeta, archivo),
                             delim_whitespace=True, encoding='latin1', header=None, comment='#')
        except Exception as e:
            print(f"No se pudo leer {archivo}: {e}")
            continue

        if subcarpeta not in representativos:
            energia = df.iloc[:, 0].values
            conteo = df.iloc[:, 1].values
            representativos[subcarpeta] = (energia, conteo, archivo)
            break  # Tomamos solo un archivo por subcarpeta

# -------------------- GRAFICA 2: Ajuste gaussiano por elemento --------------------
ejemplos_gaussiana = []

for elemento, (energia, conteo, nombre) in representativos.items():
    # Detectar picos y quitar vecinos
    picos, _ = find_peaks(conteo, prominence=0.05 * np.max(conteo))
    mascara = np.ones_like(conteo, dtype=bool)
    puntos_vecinos = 3
    for p in picos:
        ini = max(0, p - puntos_vecinos)
        fin = min(len(conteo), p + puntos_vecinos + 1)
        mascara[ini:fin] = False

    energia_sin_picos = energia[mascara]
    conteo_sin_picos = conteo[mascara]

    # Ajuste gaussiano
    try:
        p0 = [np.max(conteo_sin_picos), energia_sin_picos[np.argmax(conteo_sin_picos)],
              (energia_sin_picos[-1]-energia_sin_picos[0])/6, np.min(conteo_sin_picos)]
        popt, _ = curve_fit(gaussiana, energia_sin_picos, conteo_sin_picos, p0=p0)
        conteo_gauss = gaussiana(energia, *popt)
    except Exception as e:
        print(f"No se pudo ajustar {nombre}: {e}")
        conteo_gauss = conteo_sin_picos

    ejemplos_gaussiana.append((energia, conteo_gauss, nombre))

# Graficar ajuste gaussiano
plt.figure(figsize=(10, 6))
for energia, conteo_gauss, nombre in ejemplos_gaussiana:
    plt.plot(energia, conteo_gauss, label=f"{nombre} ajustado")

plt.savefig("2a.pdf")
plt.xlabel("Energía (keV)")
plt.ylabel("Conteo de fotones (ajustado)")
plt.title("Comparación de espectros después del ajuste gaussiano (1 por elemento)")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------- GRAFICA 3: Curvas aproximadas entre sí (1 por elemento) --------------------

# Suponiendo que cada subcarpeta corresponde a un elemento, tomamos 1 espectro por subcarpeta
representativos = {}
for energia_centrada, conteo_norm, nombre in espectros_aproximados:
    elemento = nombre.split("_")[0]  # Ajusta según cómo estén nombrados los archivos
    if elemento not in representativos:
        representativos[elemento] = (energia_centrada, conteo_norm, nombre)

plt.figure(figsize=(10, 6))
for elemento, (energia_centrada, conteo_norm, nombre) in representativos.items():
    plt.plot(energia_centrada, conteo_norm, label=f"{elemento}")

plt.savefig("2b.pdf")
plt.xlabel("Energía centrada (keV)")
plt.ylabel("Conteo normalizado")
plt.title("Curvas representativas de cada elemento aproximadas entre sí")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "2.c.pdf"))
plt.show()
