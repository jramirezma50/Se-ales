"""
01_check_fits.py

Este script es una introducción al manejo de archivos .fits.
FITS (Flexible Image Transport System) es el formato de archivo estándar
más utilizado en astronomía para guardar imágenes, datos y metadatos del telescopio.
"""

# Importamos las herramientas que vamos a utilizar:
# 'os' nos permite interactuar con el sistema de archivos (ver si un archivo existe)
import os 
# 'numpy' (alias np) es la librería matemática por excelencia de Python para trabajar con matrices
import numpy as np 
# 'matplotlib.pyplot' (alias plt) sirve para graficar o visualizar imágenes
import matplotlib.pyplot as plt 
# 'astropy.io.fits' es la herramienta especializada de astronomía para abrir archivos FITS
from astropy.io import fits # type: ignore

def inspect_fits(file_path):
    """
    Función pedagógica para abrir y explorar un archivo FITS astronómico.
    """
    print(f"--- Intentando abrir archivo FITS: {file_path} ---")
    
    # PASO 1: Verificamos si el archivo existe en nuestra computadora
    if not os.path.exists(file_path):
        print(f"⚠️ El archivo '{file_path}' no existe aún en la carpeta.")
        print("💡 En astronomía de la vida real, aquí tendrías un archivo muy pesado descargado del radiotelescopio (como el archivo generado por WSClean).")
        print("💡 Para que puedas continuar, simularé cómo se vería la estructura matemática de los datos:\n")
        
        # En una imagen astronómica o "Dirty Beam", cada píxel es simplemente un número 
        # que representa la intensidad de una señal o el peso del píxel en esa coordenada del cielo.
        # Crearemos una matriz vacía llena de números aleatorios simulando 512 píxeles de ancho por 512 de alto:
        datos_simulados = np.random.normal(0, 1, (512, 512)) # Crea ruido de fondo simulado
        
        print("📊 Dimensiones de la imagen astronómica simulada:")
        # '.shape' nos dice la forma de la matriz (Alto, Ancho)
        print(f"Matriz de datos (Data Array): {datos_simulados.shape}")
        print("-> Esto significa que tenemos una cuadrícula de 512 píxeles de alto por 512 de ancho.")
        
        # '.max()' nos dice el valor más alto dentro de esa inmensa cantidad de números
        print(f"Valor máximo (píxel más brillante): {np.max(datos_simulados):.2f}")
        return

    # PASO 2: Si el archivo existiera, así es como lo abriríamos realmente
    try:
        # Usamos fits.open() para abrir el archivo
        # Un archivo FITS es como una "cebolla", tiene múltiples "capas" o "extensiones" de datos en su interior
        with fits.open(file_path) as hdul:
            print("✅ Archivo abierto exitosamente.")
            
            # .info() nos dice qué capas (HDUs) hay dentro de la "cebolla"
            hdul.info()
            
            # Por lo general, los datos reales de la imagen astronómica están en la primera capa: índice [0]
            datos = hdul[0].data
            
            # La "cabecera" o 'header' contiene información vital: el nombre del telescopio,
            # en qué coordenadas del espacio profundo apuntó, la fecha exacta, etc.
            cabecera = hdul[0].header 
            
            # Verificamos si logramos extraer datos de imagen de esa capa
            if datos is not None:
                print(f"\n📊 Dimensiones de la imagen real obtenida del FITS: {datos.shape}")
                print(f"Tipo de dato: {datos.dtype}")
            else:
                print("La capa 0 no contiene una matriz de imágenes. Quizás los datos están en otra capa secundaria.")
                
    except Exception as e:
        # Si ocurre cualquier error abriendo el archivo, este bloque captura el error y te lo muestra
        print(f"❌ Ocurrió un error leyendo el archivo FITS: {e}")

if __name__ == "__main__":
    # Esta es la puerta de entrada de nuestro código. Cuando lo ejecutes, 
    # simulará que estamos tratando de abrir el Dirty Beam proveniente del paso de preprocesamiento de WSClean.
    
    # Inventamos la ruta al archivo FITS que algún día guardaremos allí
    archivo_prueba = "../data/raw/dirty_beam_166MHz.fits"
    
    # Llamamos a nuestra función
    inspect_fits(archivo_prueba)
