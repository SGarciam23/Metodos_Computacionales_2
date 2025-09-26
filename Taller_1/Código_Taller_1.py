import os
import numpy as np
import matplotlib.pyplot as plt

# Ruta donde están las subcarpetas con los .dat
ruta_base = r"C:\Users\usuario\OneDrive\Escritorio\Universidad\Materias\Metodos Computacionales 2\mammography_spectra"

plt.figure(figsize=(10, 6))

# Recorrer cada subcarpeta (alta, media, baja energía)
for subcarpeta in os.listdir(ruta_base):
    ruta_subcarpeta = os.path.join(ruta_base, subcarpeta)
    if os.path.isdir(ruta_subcarpeta):
        energias_lista = []
        conteos_lista = []

        # Recorrer cada archivo .dat de la subcarpeta
        for archivo in os.listdir(ruta_subcarpeta):
            if archivo.endswith(".dat"):
                ruta_archivo = os.path.join(ruta_subcarpeta, archivo)
                try:
                    with open(ruta_archivo, encoding='latin1') as f:
                        lineas = f.readlines()
                        datos = np.array([
                            list(map(float, linea.strip().split()))
                            for linea in lineas if linea.strip() and not linea.startswith('#')
                        ])

                    energia = datos[:, 0]
                    conteo = datos[:, 1]

                    energias_lista.append(energia)
                    conteos_lista.append(conteo)

                except Exception as e:
                    print(f"Error con {ruta_archivo}: {e}")

        if not energias_lista:
            continue

        # 📌 Usar el rango más amplio posible
        energia_min = min(energia.min() for energia in energias_lista)
        energia_max = max(energia.max() for energia in energias_lista)
        energia_comun = np.linspace(energia_min, energia_max, 800)  # más puntos para más detalle

        # Interpolar todas las curvas al eje común (fuera de su rango -> NaN)
        conteos_interp = []
        for energia, conteo in zip(energias_lista, conteos_lista):
            conteos_interp.append(np.interp(energia_comun, energia, conteo, left=np.nan, right=np.nan))

        # Convertir a array y promediar ignorando NaN
        conteos_array = np.array(conteos_interp)
        conteo_ponderado = np.nanmean(conteos_array, axis=0)

        # Graficar el promedio de la carpeta
        plt.plot(energia_comun, conteo_ponderado, label=subcarpeta)

        # Mostrar información
        print(f"{subcarpeta} → Energía min: {energia_min:.2f} keV, max: {energia_max:.2f} keV")

# Ajustes de la gráfica
plt.xlabel("Energía (keV)")
plt.ylabel("Conteo de fotones (promedio ponderado)")
plt.title("Espectros promediados en función de su energía")
plt.legend()
plt.savefig("1.a.pdf")
plt.tight_layout()
plt.show()
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
