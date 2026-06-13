"""
PI-AstroDeconv: Script de Evaluación y Métricas

Calcula métricas astronómicas de calidad de deconvolución:
- Power Spectrum (específico para 21cm)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE/RMSE (Error Absoluto/Cuadrático)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft2, fftshift
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tensorflow as tf
from astropy.io import fits
from pathlib import Path


def compute_power_spectrum(image):
    """
    Calcula power spectrum 2D de la imagen.
    
    El power spectrum es crucial en astronomía 21cm porque:
    - Representa la distribución de energía en frecuencias espaciales
    - Debe preservar estructuras (bajo no debe estar amplificado)
    - Usado para HI (Neutral Hydrogen) mapping
    
    Args:
        image: Array 2D
    
    Returns:
        power_spectrum: Espectro de potencia 1D (radial)
        frequencies: Frecuencias espaciales (radial)
    """
    L = image.shape[0]
    
    # Aplicar FFT
    fft_image = fft2(image)
    power = np.abs(fft_image) ** 2
    
    # Centrar (shift DC al centro)
    power = fftshift(power)
    
    # Crear grid de frecuencias
    freq = np.fft.fftfreq(L)
    freq_shifted = fftshift(freq)
    
    # Convertir a coordenadas radiales
    fx = freq_shifted
    fy = freq_shifted
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    r_grid = np.sqrt(fx_grid**2 + fy_grid**2)
    
    # Agrupar por radio
    r_unique = np.unique(r_grid)
    power_radial = []
    
    for r in r_unique:
        mask = (r_grid == r)
        power_radial.append(np.mean(power[mask]))
    
    return np.array(power_radial), r_unique


class AstroDeconvEvaluator:
    """
    Evaluador de modelos astronómicos con métricas especializadas.
    """
    
    def __init__(self, model_path=None):
        """
        Inicializa el evaluador.
        
        Args:
            model_path: Ruta del modelo entrenado (opcional)
        """
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Carga un modelo entrenado.
        
        Args:
            model_path: Ruta del archivo .h5
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Modelo cargado: {model_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
    
    def evaluate_metrics(self, clean_truth, clean_predicted, name=""):
        """
        Calcula todas las métricas de evaluación.
        
        Args:
            clean_truth: Imagen limpia verdadera (Ground Truth)
            clean_predicted: Imagen limpia predicha por el modelo
            name: Nombre para el reporte
        
        Returns:
            Diccionario con todas las métricas
        """
        print(f"\n{'='*60}")
        print(f"📊 MÉTRICAS DE EVALUACIÓN: {name}")
        print(f"{'='*60}")
        
        # Normalizar
        clean_truth = self._normalize(clean_truth)
        clean_predicted = self._normalize(clean_predicted)
        
        # 1. MAE (Mean Absolute Error)
        mae = np.mean(np.abs(clean_truth - clean_predicted))
        print(f"\n1️⃣  MAE (Error Absoluto Medio)")
        print(f"    Valor: {mae:.6f}")
        print(f"    → Qué significa: Diferencia promedio entre píxeles")
        
        # 2. RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((clean_truth - clean_predicted) ** 2))
        print(f"\n2️⃣  RMSE (Error Cuadrático Medio)")
        print(f"    Valor: {rmse:.6f}")
        print(f"    → Qué significa: Penaliza errores grandes más que MAE")
        
        # 3. PSNR (Peak Signal-to-Noise Ratio)
        try:
            psnr = peak_signal_noise_ratio(
                clean_truth, 
                clean_predicted, 
                data_range=1.0
            )
            print(f"\n3️⃣  PSNR (Peak Signal-to-Noise Ratio)")
            print(f"    Valor: {psnr:.2f} dB")
            print(f"    → Qué significa: Mejor es mayor (típico >20 dB es bueno)")
        except Exception as e:
            print(f"\n3️⃣  PSNR: No calculable ({e})")
            psnr = None
        
        # 4. SSIM (Structural Similarity Index)
        try:
            ssim = structural_similarity(
                clean_truth, 
                clean_predicted, 
                data_range=1.0
            )
            print(f"\n4️⃣  SSIM (Structural Similarity)")
            print(f"    Valor: {ssim:.4f}")
            print(f"    → Qué significa: Similitud estructural (0-1, mejor >0.8)")
        except Exception as e:
            print(f"\n4️⃣  SSIM: No calculable ({e})")
            ssim = None
        
        # 5. Power Spectrum Error (Astronómico)
        ps_truth, freq_truth = compute_power_spectrum(clean_truth)
        ps_pred, _ = compute_power_spectrum(clean_predicted)
        
        ps_error = np.mean(np.abs(ps_truth - ps_pred))
        print(f"\n5️⃣  POWER SPECTRUM ERROR (21cm específico)")
        print(f"    Valor: {ps_error:.6e}")
        print(f"    → Qué significa: Error en distribución de energía")
        
        print(f"\n{'='*60}\n")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'psnr': psnr,
            'ssim': ssim,
            'ps_error': ps_error,
            'ps_truth': ps_truth,
            'ps_pred': ps_pred,
            'frequencies': freq_truth
        }
    
    def visualize_deconvolution(self, dirty_image, clean_truth, 
                                clean_predicted, save_path=None):
        """
        Visualiza antes/después de deconvolución en 4 paneles.
        
        Args:
            dirty_image: Imagen sucia observada
            clean_truth: Cielo limpio verdadero
            clean_predicted: Cielo limpio predicho
            save_path: Ruta para guardar (opcional)
        
        Returns:
            Figura matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        
        # Normalizar
        dirty_image = self._normalize(dirty_image)
        clean_truth = self._normalize(clean_truth)
        clean_predicted = self._normalize(clean_predicted)
        
        # Panel 1: Cielo Sucio (Entrada)
        im0 = axes[0, 0].imshow(dirty_image, cmap='inferno', origin='lower')
        axes[0, 0].set_title('Cielo Sucio (Dirty Image - Observado)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], label='Intensidad')
        
        # Panel 2: Cielo Limpio Real (Ground Truth)
        im1 = axes[0, 1].imshow(clean_truth, cmap='inferno', origin='lower')
        axes[0, 1].set_title('Cielo Limpio Real (Ground Truth)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Intensidad')
        
        # Panel 3: Cielo Limpio Predicho (Modelo)
        im2 = axes[1, 0].imshow(clean_predicted, cmap='inferno', origin='lower')
        axes[1, 0].set_title('Cielo Limpio Predicho (Modelo Entrenado)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], label='Intensidad')
        
        # Panel 4: Error (Ground Truth - Predicción)
        error = np.abs(clean_truth - clean_predicted)
        im3 = axes[1, 1].imshow(error, cmap='hot', origin='lower')
        mae_val = np.mean(error)
        axes[1, 1].set_title(f'Error Absoluto (MAE={mae_val:.6f})', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], label='Error')
        
        plt.tight_layout()
        
        # Guardar si se especifica
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualización guardada: {save_path}")
        
        return fig
    
    def plot_power_spectrum(self, clean_truth, clean_predicted, save_path=None):
        """
        Grafica power spectrum en escala logarítmica.
        
        Crucial para astronomía 21cm: muestra si se preservan estructuras.
        
        Args:
            clean_truth: Ground truth
            clean_predicted: Predicción del modelo
            save_path: Ruta para guardar (opcional)
        
        Returns:
            Figura matplotlib
        """
        ps_truth, freq = compute_power_spectrum(clean_truth)
        ps_pred, _ = compute_power_spectrum(clean_predicted)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Graficar en escala log-log
        ax.loglog(freq, ps_truth, 'o-', label='Ground Truth', 
                 linewidth=2, markersize=6, alpha=0.7)
        ax.loglog(freq, ps_pred, 's-', label='Modelo Predicho', 
                 linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Frecuencia Espacial', fontsize=12, fontweight='bold')
        ax.set_ylabel('Power Spectrum', fontsize=12, fontweight='bold')
        ax.set_title('Power Spectrum: Ground Truth vs Modelo\n(21cm HI Mapping)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Guardar si se especifica
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Power spectrum guardado: {save_path}")
        
        return fig
    
    @staticmethod
    def _normalize(data):
        """
        Normaliza array entre 0 y 1.
        
        Args:
            data: Array a normalizar
        
        Returns:
            Array normalizado
        """
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        return data


def main():
    """
    Pipeline completo de evaluación.
    """
    print("\n" + "="*60)
    print("PI-AstroDeconv: Pipeline de Evaluación")
    print("="*60)
    
    data_path = Path('data/raw/')
    output_path = Path('models/')
    output_path.mkdir(exist_ok=True)
    
    # Crear evaluador
    evaluator = AstroDeconvEvaluator()
    
    # Cargar datos
    print("\n📂 Cargando datos FITS...")
    try:
        with fits.open(data_path / 'clean_sky.fits') as hdul:
            clean_truth = hdul[0].data.astype(np.float32)
        with fits.open(data_path / 'dirty_sky.fits') as hdul:
            dirty_image = hdul[0].data.astype(np.float32)
        print("✅ Datos cargados exitosamente")
    except FileNotFoundError as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # Simular predicción (en real: usar modelo entrenado)
    print("\n🤖 Generando predicción...")
    np.random.seed(42)
    clean_predicted = clean_truth + np.random.normal(0, 0.05, clean_truth.shape)
    clean_predicted = np.clip(clean_predicted, 0, 1)
    print("✅ Predicción generada")
    
    # Evaluar
    print("\n" + "-"*60)
    print("FASE 1: MÉTRICAS DE EVALUACIÓN")
    print("-"*60)
    metrics = evaluator.evaluate_metrics(clean_truth, clean_predicted, 
                                         name="Test Set")
    
    # Visualizar deconvolución
    print("\n" + "-"*60)
    print("FASE 2: VISUALIZACIÓN COMPARATIVA")
    print("-"*60)
    fig1 = evaluator.visualize_deconvolution(
        dirty_image, clean_truth, clean_predicted,
        save_path=output_path / 'deconvolution_comparison.png'
    )
    
    # Graficar power spectrum
    print("\n" + "-"*60)
    print("FASE 3: ANÁLISIS POWER SPECTRUM (21cm)")
    print("-"*60)
    fig2 = evaluator.plot_power_spectrum(
        clean_truth, clean_predicted,
        save_path=output_path / 'power_spectrum_analysis.png'
    )
    
    print("\n" + "="*60)
    print("✅ Evaluación completada")
    print("   Archivos generados en: models/")
    print("="*60)
    
    return True


if __name__ == "__main__":
    main()
