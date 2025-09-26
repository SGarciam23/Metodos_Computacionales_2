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


#Punto 2a, intento 2 

# punto 2

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Ruta base donde están las carpetas W, Rh, Mo
ruta_base = r"/content/mammography_spectra"

# Diccionario para guardar ejemplos por elemento
ejemplos_remocion_por_elemento = {"W": [], "Rh": [], "Mo": []}

# Parámetro para puntos vecinos
puntos_vecinos = 3
prom_rel = 0.05 # Usar el mismo umbral relativo para detección de picos

for subcarpeta in os.listdir(ruta_base):
    ruta_subcarpeta = os.path.join(ruta_base, subcarpeta)
    if not os.path.isdir(ruta_subcarpeta):
        continue

    # Detectar elemento por nombre de carpeta
    if "W" in subcarpeta and len(ejemplos_remocion_por_elemento["W"]) < 2:
        elemento = "W"
    elif "Rh" in subcarpeta and len(ejemplos_remocion_por_elemento["Rh"]) < 2:
        elemento = "Rh"
    elif "Mo" in subcarpeta and len(ejemplos_remocion_por_elemento["Mo"]) < 2:
        elemento = "Mo"
    else:
        continue # Saltar si ya tenemos 2 ejemplos para este elemento o no es uno de los elementos principales


    for archivo in os.listdir(ruta_subcarpeta):
        if not archivo.endswith(".dat"):
            continue

        # Leer datos
        # Specify the encoding as 'latin1' to handle the character encoding issue
        try:
            datos = np.loadtxt(os.path.join(ruta_subcarpeta, archivo), encoding='latin1')
            energia, conteo = datos[:, 0], datos[:, 1]

            # Detectar picos con umbral relativo
            picos, _ = find_peaks(conteo, prominence=prom_rel * np.max(conteo))

            # Crear máscara para eliminar los picos y sus vecinos
            mascara = np.ones_like(conteo, dtype=bool)
            for p in picos:
                ini = max(0, p - puntos_vecinos)
                fin = min(len(conteo), p + puntos_vecinos + 1)
                mascara[ini:fin] = False

            energia_sin_picos = energia[mascara]
            conteo_sin_picos = conteo[mascara]

            # Guardar ejemplo si aún no tenemos 2 para este elemento
            if len(ejemplos_remocion_por_elemento[elemento]) < 2:
                 ejemplos_remocion_por_elemento[elemento].append((energia, conteo, energia_sin_picos, conteo_sin_picos, archivo))


        except Exception as e:
            print(f"Error processing {archivo}: {e}")

# --- Graficar ejemplos por elemento ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
orden = ["W", "Rh", "Mo"]
titulos = {"W": "Tungsteno (W)", "Rh": "Rodio (Rh)", "Mo": "Molibdeno (Mo)"}

for i, elem in enumerate(orden):
    ax = axs[i]
    if not ejemplos_remocion_por_elemento[elem]:
        ax.set_title(f"{titulos[elem]} — sin ejemplos")
        ax.set_xlabel("Energía (keV)")
        ax.set_ylabel("Conteo de fotones")
        continue

    for energia, conteo, e_sin, c_sin, nombre in ejemplos_remocion_por_elemento[elem]:
        ax.plot(energia, conteo, alpha=0.6, label=f"{nombre} original")
        ax.plot(e_sin, c_sin, 'o', markersize=3, label=f"{nombre} sin picos")

    ax.set_title(f"Ejemplos de espectros con picos removidos — {titulos[elem]}")
    ax.set_xlabel("Energía (keV)")
    ax.set_ylabel("Conteo de fotones")
    ax.legend(ncol=2, fontsize=8)


plt.tight_layout()
plt.savefig("2.a.pdf")
plt.show()


#opcion #2, 2b

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Ruta base
ruta_base = r"/content/mammography_spectra"

# Diccionario para guardar ejemplos por elemento
ejemplos_aprox_por_elemento = {"W": [], "Rh": [], "Mo": []}

# Rango de puntos eliminados alrededor del pico
puntos_vecinos = 3
prom_rel = 0.05 # Usar el mismo umbral relativo para detección de picos


for subcarpeta in os.listdir(ruta_base):
    ruta_subcarpeta = os.path.join(ruta_base, subcarpeta)
    if not os.path.isdir(ruta_subcarpeta):
        continue

    # Detectar elemento por nombre de carpeta
    if "W" in subcarpeta and len(ejemplos_aprox_por_elemento["W"]) < 2:
        elemento = "W"
    elif "Rh" in subcarpeta and len(ejemplos_aprox_por_elemento["Rh"]) < 2:
        elemento = "Rh"
    elif "Mo" in subcarpeta and len(ejemplos_aprox_por_elemento["Mo"]) < 2:
        elemento = "Mo"
    else:
        continue # Saltar si ya tenemos 2 ejemplos para este elemento o no es uno de los elementos principales


    for archivo in os.listdir(ruta_subcarpeta):
        if not archivo.endswith(".dat"):
            continue

        # Leer datos
        # Specify the encoding as 'latin1' to handle the character encoding issue
        try:
            datos = np.loadtxt(os.path.join(ruta_subcarpeta, archivo), encoding='latin1')
            energia, conteo = datos[:, 0], datos[:, 1]

            # --- Limpiar picos ---
            picos, _ = find_peaks(conteo, prominence=prom_rel * np.max(conteo))
            mascara = np.ones_like(conteo, dtype=bool)
            for p in picos:
                ini = max(0, p - puntos_vecinos)
                fin = min(len(conteo), p + puntos_vecinos + 1)
                mascara[ini:fin] = False

            energia_sin_picos = energia[mascara]
            conteo_sin_picos = conteo[mascara]

            # --- Aproximar continuo ---
            kind_interp = "cubic" if len(energia_sin_picos) >= 4 else "linear"
            interp_func = interp1d(energia_sin_picos, conteo_sin_picos, kind=kind_interp, fill_value="extrapolate")
            continuo_aprox = interp_func(energia)


            # Guardar dos ejemplos para graficar por elemento
            if len(ejemplos_aprox_por_elemento[elemento]) < 2:
                ejemplos_aprox_por_elemento[elemento].append((energia, conteo, continuo_aprox, archivo))

        except Exception as e:
            print(f"Error processing {archivo}: {e}")

# --- Graficar ejemplos ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
orden = ["W", "Rh", "Mo"]
titulos = {"W": "Tungsteno (W)", "Rh": "Rodio (Rh)", "Mo": "Molibdeno (Mo)"}

for i, elem in enumerate(orden):
    ax = axs[i]
    if not ejemplos_aprox_por_elemento[elem]:
        ax.set_title(f"{titulos[elem]} — sin ejemplos")
        ax.set_xlabel("Energía (keV)")
        ax.set_ylabel("Conteo de fotones")
        continue

    for energia, conteo, continuo, nombre in ejemplos_aprox_por_elemento[elem]:
        ax.plot(energia, conteo, alpha=0.5, label=f"{nombre} original")
        ax.plot(energia, continuo, '--', label=f"{nombre} continuo aprox")

    ax.set_title(f"Aproximación del continuo con picos eliminados — {titulos[elem]}")
    ax.set_xlabel("Energía (keV)")
    ax.set_ylabel("Conteo de fotones")
    ax.legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.savefig("2.b.pdf")
plt.show()



#3a

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Ruta base
ruta_base = r"/content/mammography_spectra"

# Parámetros de limpieza
puntos_vecinos = 3
prom_rel = 0.05
margen_zoom_keV = 1.0

# Diccionario para almacenar datos por elemento
residuales = {"W": [], "Rh": [], "Mo": []}
rangos_picos = {"W": [], "Rh": [], "Mo": []}

# Procesar datos
for subcarpeta in os.listdir(ruta_base):
    ruta_subcarpeta = os.path.join(ruta_base, subcarpeta)
    if not os.path.isdir(ruta_subcarpeta):
        continue

    # Detectar elemento por nombre de carpeta
    if "W" in subcarpeta:
        elemento = "W"
    elif "Rh" in subcarpeta:
        elemento = "Rh"
    elif "Mo" in subcarpeta:
        elemento = "Mo"
    else:
        continue

    for archivo in os.listdir(ruta_subcarpeta):
        if not archivo.endswith(".dat"):
            continue

        # Leer datos originales
        datos = np.loadtxt(os.path.join(ruta_subcarpeta, archivo), encoding='latin1')
        energia, conteo = datos[:, 0], datos[:, 1]

        # --- Limpiar picos ---
        picos, _ = find_peaks(conteo, prominence=prom_rel * np.max(conteo))
        mascara = np.ones_like(conteo, dtype=bool)
        for p in picos:
            ini = max(0, p - puntos_vecinos)
            fin = min(len(conteo), p + puntos_vecinos + 1)
            mascara[ini:fin] = False

        energia_sin_picos = energia[mascara]
        conteo_sin_picos = conteo[mascara]

        # --- Aproximar continuo ---
        kind_interp = "cubic" if len(energia_sin_picos) >= 4 else "linear"
        interp_func = interp1d(energia_sin_picos, conteo_sin_picos, kind=kind_interp, fill_value="extrapolate")
        continuo_aprox = interp_func(energia)

        # --- Calcular residual ---
        residual = conteo - continuo_aprox

        # --- Detectar picos en residual ---
        pidx, _ = find_peaks(residual, prominence=0.01 * np.max(residual) if np.max(residual) > 0 else 1e9)
        if len(pidx) == 0:
            continue

        emin = energia[pidx].min() - margen_zoom_keV
        emax = energia[pidx].max() + margen_zoom_keV

        residuales[elemento].append((energia, residual, archivo))
        rangos_picos[elemento].append((emin, emax))

# --- Graficar ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
orden = ["W", "Rh", "Mo"]
titulos = {"W": "Tungsteno (W)", "Rh": "Rodio (Rh)", "Mo": "Molibdeno (Mo)"}

for i, elem in enumerate(orden):
    ax = axs[i]
    if len(residuales[elem]) == 0:
        ax.set_title(f"{titulos[elem]} — sin picos detectables")
        ax.set_xlabel("Energía (keV)")
        ax.set_ylabel("Residual (picos)")
        continue

    # Zoom global para ese elemento
    emin_global = min(e for e, _ in rangos_picos[elem])
    emax_global = max(e for _, e in rangos_picos[elem])

    for energia, residual, nombre in residuales[elem]:
        mask = (energia >= emin_global) & (energia <= emax_global)
        ax.plot(energia[mask], residual[mask], alpha=0.7, label=nombre.replace(".dat", ""))

    ax.set_title(f"Picos aislados — {titulos[elem]}")
    ax.set_xlim(emin_global, emax_global)
    ax.set_xlabel("Energía (keV)")
    ax.set_ylabel("Residual (cuentas)")
    ax.legend(ncol=4, fontsize=8)

plt.tight_layout()
plt.savefig("3.a.pdf")
plt.show()
