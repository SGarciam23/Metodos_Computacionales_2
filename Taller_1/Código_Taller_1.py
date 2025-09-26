import os 
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

#---------
# PUNTO 1
#---------

# Ruta donde están las subcarpetas con los .dat
ruta_base = r"C:\Users\usuario\OneDrive\Escritorio\Universidad\Materias\Metodos Computacionales 2\mammography_spectra"

plt.figure(figsize=(10, 6))

# Diccionario para acumular datos por elemento
datos_por_elemento = {"Mo": [], "Rh": [], "W": []}
kv_por_elemento = {"Mo": [], "Rh": [], "W": []}  # para guardar voltajes

# Recorrer cada subcarpeta (alta, media, baja energía)
for subcarpeta in os.listdir(ruta_base):
    ruta_subcarpeta = os.path.join(ruta_base, subcarpeta)
    if os.path.isdir(ruta_subcarpeta):
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

                    # Detectar a qué elemento pertenece
                    if "Mo" in archivo:
                        datos_por_elemento["Mo"].append((energia, conteo))
                        kv = int("".join([c for c in archivo if c.isdigit()]))
                        kv_por_elemento["Mo"].append(kv)
                    elif "Rh" in archivo:
                        datos_por_elemento["Rh"].append((energia, conteo))
                        kv = int("".join([c for c in archivo if c.isdigit()]))
                        kv_por_elemento["Rh"].append(kv)
                    elif "W" in archivo:
                        datos_por_elemento["W"].append((energia, conteo))
                        kv = int("".join([c for c in archivo if c.isdigit()]))
                        kv_por_elemento["W"].append(kv)

                except Exception as e:
                    print(f"Error con {ruta_archivo}: {e}")

# Graficar un espectro promedio para cada elemento con metadatos en el label
for elemento, espectros in datos_por_elemento.items():
    if espectros:
        # Unir rangos y crear eje común
        energia_min = min(e.min() for e, _ in espectros)
        energia_max = max(e.max() for e, _ in espectros)
        energia_comun = np.linspace(energia_min, energia_max, 800)

        # Interpolar y promediar
        conteos_interp = []
        for energia, conteo in espectros:
            conteos_interp.append(np.interp(energia_comun, energia, conteo, left=np.nan, right=np.nan))
        conteos_array = np.array(conteos_interp)
        conteo_promedio = np.nanmean(conteos_array, axis=0)

        # Calcular valores representativos
        conteo_max = np.nanmax(conteo_promedio)
        kv_medio = np.mean(kv_por_elemento[elemento]) if kv_por_elemento[elemento] else np.nan

        # Label detallado estilo artículo científico
        label = (f"{elemento} | Energía: {energia_min:.1f}-{energia_max:.1f} keV | "
                 f"Vtubo ≈ {kv_medio:.0f} kV | "
                 f"Conteo máx: {conteo_max:.0f}")

        # Graficar solo 1 curva por elemento
        plt.plot(energia_comun, conteo_promedio, label=label, linewidth=2)

# Ajustes de la gráfica
plt.xlabel("Energía (keV)")
plt.ylabel("Conteo de fotones (promedio)")
plt.title("Espectros característicos por elemento del ánodo")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("1.pdf", bbox_inches="tight", pad_inches=0.1)

#---------
# PUNTO 2
#---------

#Punto 2a, intento 2 

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

#opcion #2, 2b

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

# 2.c

resultados_continuo = {"W": [], "Rh": [], "Mo": []}

for elem in ejemplos_aprox_por_elemento:
  for energia, conteo, continuo, nombre in ejemplos_aprox_por_elemento[elem]:
    max_valor = np.max(continuo)
    idx_max = np.argmax(continuo)
    energia_max = energia[idx_max]
    mitad = max_valor / 2
    indices_arriba = np.where(continuo >= mitad)[0]
    if len(indices_arriba) >= 2:
      fwhm = energia[indices_arriba[-1]] - energia[indices_arriba[0]]
    else:
      fwhm = np.nan
      
    voltaje_match = re.search(r"(\d+)[kK]V", nombre)

    if voltaje_match:
      voltaje = int(voltaje_match.group(1))
    else:
      voltaje = np.nan

    resultados_continuo[elem].append((voltaje, max_valor, energia_max, fwhm))

# --- Graficar resultados ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
colores = {"W": "tab:blue", "Rh": "tab:green", "Mo": "tab:red"}

for i, variable in enumerate(["Máximo", "Energía del máximo", "FWHM"]):
  ax = axs[i // 2][i % 2]
  for elem in resultados_continuo:
    datos = np.array(resultados_continuo[elem])
    if datos.size == 0:
      continue
    voltajes = datos[:, 0]
    valores = datos[:, i + 1]
  ax.plot(voltajes, valores, 'o-', label=elem, color=colores[elem])
  ax.set_title(f"{variable} vs Voltaje del tubo")
  ax.set_xlabel("Voltaje (kV)")
  ax.set_ylabel(variable)
  ax.legend()

ax = axs[1][1]
for elem in resultados_continuo:
  datos = np.array(resultados_continuo[elem])
  if datos.size == 0:
    continue
  energia_max = datos[:, 2]
  max_valor = datos[:, 1]
  ax.plot(energia_max, max_valor, 's-', label=elem, color=colores[elem])
ax.set_title("Máximo del continuo vs Energía del máximo")
ax.set_xlabel("Energía (keV)")
ax.set_ylabel("Máximo del continuo")
ax.legend()

plt.tight_layout()
plt.savefig("2.c.pdf")

#---------
# PUNTO 3
#---------

#3a

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

# 3.b

def gauss(x, A, mu, sigma):
  return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

resultados_ajuste = {"W": [], "Rh": [], "Mo": []}

for elemento in orden:
  for energia, residual, nombre in residuales[elemento]:
    pidx, _ = find_peaks(residual, prominence=0.01 * np.max(residual))
    if len(pidx) == 0:
      continue
    idx_max = pidx[np.argmax(residual[pidx])]
    mu_est = energia[idx_max]
    A_est = residual[idx_max]
    sigma_est = 0.2

    ventana = (energia > mu_est - 1.0) & (energia < mu_est + 1.0)
    x_fit = energia[ventana]
    y_fit = residual[ventana]

    if len(x_fit) < 3 or np.max(y_fit) < 1e-3:
      continue 

    try:
      popt, _ = curve_fit(gauss, x_fit, y_fit, p0=[A_est, mu_est, sigma_est], maxfev=5000)
      A, mu, sigma = popt
      fwhm = 2.355 * abs(sigma)
    except Exception as e:
      print(f"⚠️ Falló el ajuste en {nombre}: {e}")
      continue

    match = re.search(r"(\d+)\s*kV", nombre)
    
    if match:
      voltaje = int(match.group(1))
    else:
      print(f"⚠️ No se pudo extraer voltaje de: {nombre}")
      continue

    resultados_ajuste[elemento].append((voltaje, A, fwhm))

# --- Graficar resultados ---
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for elemento in orden:
  datos = np.array(resultados_ajuste[elemento])
  if len(datos) == 0:
    continue

  datos = datos[np.argsort(datos[:, 0])]
  voltajes = datos[:, 0]
  alturas = datos[:, 1]
  fwhms = datos[:, 2]

  axs[0].plot(voltajes, alturas, 'o-', label=titulos[elemento])
  axs[1].plot(voltajes, fwhms, 's--', label=titulos[elemento])

axs[0].set_title("Altura del pico vs Voltaje del tubo")
axs[0].set_xlabel("Voltaje (kV)")
axs[0].set_ylabel("Altura del pico (cuentas)")
axs[0].legend()

axs[1].set_title("Ancho a media altura (FWHM) vs Voltaje del tubo")
axs[1].set_xlabel("Voltaje (kV)")
axs[1].set_ylabel("FWHM (keV)")
axs[1].legend()

plt.tight_layout()
plt.savefig("3.b.pdf")
