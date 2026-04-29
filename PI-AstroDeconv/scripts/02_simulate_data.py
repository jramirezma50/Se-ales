"""
02_simulate_data.py

Este script fabrica nuestros propios datos astronómicos controlados.
Generaremos un "Cielo Limpio", fabricaremos el defecto del telescopio ("Dirty Beam")
y los mezclaremos matemáticamente para obtener el "Cielo Sucio" (Dirty Image).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from astropy.io import fits # type: ignore

def create_clean_sky(size=512):
    """
    Crea una imagen de un cielo negro con algunas fuentes luminosas brillantes.
    """
    sky = np.zeros((size, size))
    # Colocamos 20 "galaxias" al azar
    num_sources = 20
    np.random.seed(42) # Fijamos la semilla para que siempre nos dé el mismo cielo
    
    for _ in range(num_sources):
        x = np.random.randint(20, size-20)
        y = np.random.randint(20, size-20)
        brightness = np.random.uniform(10, 30)
        # Simulamos que cada fuente tiene un núcleo muy brillante y bordes suaves
        sky[x-1:x+2, y-1:y+2] += brightness * 0.5
        sky[x, y] += brightness
    return sky

def create_dirty_beam(size=512):
    """
    Simula el defecto del telescopio (Point Spread Function / Dirty Beam).
    Un telescopio de interferometría suele tener un pico central fuerte y ondas a su alrededor (sidelobes).
    """
    # Creamos un plano de coordenadas matemáticas
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Usamos una función Sinc combinada con un decaimiento gaussiano 
    # Esto imita visualmente el patrón de interferencia real de un arreglo de antenas
    beam = np.sin(R * 4) / (R * 4 + 1e-5) * np.exp(-R**2 / 10)
    
    # En astronomía, la suma de energía del beam debe normalizarse a 1
    beam /= np.sum(beam) 
    return beam

def save_to_fits(data, filename):
    """Guarda la matriz matemática como un archivo astronómico .fits profesional."""
    # En FITS, la información principal se guarda en el 'PrimaryHDU'
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filename, overwrite=True)
    print(f"💾 Guardado: {filename}")

def main():
    print("🔭 Iniciando simulación de nuestro pequeño universo...\n")
    
    # 1. Fabricamos la verdad (lo que queremos que adivine la IA)
    clean_sky = create_clean_sky()
    
    # 2. Fabricamos el problema (el patrón de las antenas)
    dirty_beam = create_dirty_beam()
    
    # 3. Fabricamos la observación real (Convolucionamos la verdad con el problema)
    print("⏳ Aplicando el efecto del telescopio (convolución FFT)...")
    # Nota: fftconvolve usa la Transformada Rápida de Fourier, ¡exactamente lo que menciona el paper!
    dirty_sky = fftconvolve(clean_sky, dirty_beam, mode='same')
    
    # 4. Guardamos todo en nuestra carpeta de datos crudos
    # Usamos la ruta asumiendo que ejecutamos el script desde la raíz del proyecto
    raw_dir = "data/raw" 
    os.makedirs(raw_dir, exist_ok=True)
    
    save_to_fits(clean_sky, os.path.join(raw_dir, "clean_sky.fits"))
    save_to_fits(dirty_beam, os.path.join(raw_dir, "dirty_beam.fits"))
    save_to_fits(dirty_sky, os.path.join(raw_dir, "dirty_sky.fits"))
    
    print("\n✅ Datos FITS generados exitosamente. ¡Estamos listos para la Inteligencia Artificial!")
    
    # 5. Visualización para entender el reto
    plt.style.use('dark_background')
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("1. Cielo Limpio\n(La Verdad Oculta)")
    plt.imshow(clean_sky, cmap='inferno', origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(1, 3, 2)
    plt.title("2. Dirty Beam\n(El Defecto del Telescopio)")
    # Hacemos zoom al centro de la mancha para que notes los anillos sutiles
    centro = 512 // 2
    zoom = 60
    plt.imshow(dirty_beam[centro-zoom:centro+zoom, centro-zoom:centro+zoom], cmap='viridis', origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(1, 3, 3)
    plt.title("3. Dirty Sky\n(Lo que entregaría WSClean)")
    plt.imshow(dirty_sky, cmap='inferno', origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    imagen_path = os.path.join(raw_dir, "comparacion_simulacion.png")
    plt.savefig(imagen_path, dpi=150)
    print(f"🖼️ Te he guardado también una imagen comparativa en: {imagen_path}")
    
    print("⏳ Abriendo visualización gráfica... (Cierra la ventana para terminar el script)")
    plt.show()

if __name__ == "__main__":
    # Usamos dirname para que funcione desde cualquier terminal
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
