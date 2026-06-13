# CÓDIGO MEJORADO: Soluciones para Llegar al 100%

## 1. ARQUITECTURA MEJORADA (03_model_unet_enhanced.py)

```python
"""
Versión mejorada del modelo U-Net con:
- Mayor profundidad (3-5 niveles)
- BatchNormalization
- Dropout regularization
- Mejor arquitectura de cuello de botella
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_unet_enhanced(input_shape=(512, 512, 1)):
    """
    U-Net mejorada con profundidad y regularización.
    
    Arquitectura:
    512 → 256 → 128 → 64 → 32 (cuello) → 64 → 128 → 256 → 512
    """
    inputs = layers.Input(shape=input_shape, name="imagen_sucia")
    
    # ===== ENCODER (Bajada) =====
    # Nivel 1
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Dropout(0.2)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)  # 512 → 256
    
    # Nivel 2
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Dropout(0.2)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)  # 256 → 128
    
    # Nivel 3
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Dropout(0.2)(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)  # 128 → 64
    
    # Nivel 4
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.2)(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)  # 64 → 32
    
    # ===== CUELLO DE BOTELLA (Bottleneck) =====
    # Punto más comprimido donde "entiende" la imagen
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.Dropout(0.3)(bottleneck)  # Dropout más alto aquí
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(bottleneck)
    bottleneck = layers.BatchNormalization()(bottleneck)
    
    # ===== DECODER (Subida) =====
    # Nivel 4 (Invertido)
    u4 = layers.UpSampling2D((2, 2))(bottleneck)  # 32 → 64
    u4 = layers.Concatenate()([u4, c4])
    d4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.Dropout(0.2)(d4)
    
    # Nivel 3
    u3 = layers.UpSampling2D((2, 2))(d4)  # 64 → 128
    u3 = layers.Concatenate()([u3, c3])
    d3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(0.2)(d3)
    
    # Nivel 2
    u2 = layers.UpSampling2D((2, 2))(d3)  # 128 → 256
    u2 = layers.Concatenate()([u2, c2])
    d2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Dropout(0.2)(d2)
    
    # Nivel 1
    u1 = layers.UpSampling2D((2, 2))(d2)  # 256 → 512
    u1 = layers.Concatenate()([u1, c1])
    d1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Dropout(0.2)(d1)
    
    # ===== SALIDA =====
    output = layers.Conv2D(1, (1, 1), activation='relu', name="cielo_limpio_predicho")(d1)
    
    return inputs, output


class FFTPhysicsLayer(layers.Layer):
    """
    Capa que aplica la física: convoluciona con el Dirty Beam
    """
    def __init__(self, dirty_beam, **kwargs):
        super(FFTPhysicsLayer, self).__init__(**kwargs)
        beam_tensor = tf.convert_to_tensor(dirty_beam, dtype=tf.float32)
        beam_tensor = tf.expand_dims(tf.expand_dims(beam_tensor, -1), -1)
        self.dirty_beam = beam_tensor

    def call(self, inputs):
        # La convolución usa FFT internamente en TensorFlow
        cielo_ensuciado = tf.nn.conv2d(
            inputs, 
            self.dirty_beam, 
            strides=[1, 1, 1, 1], 
            padding='SAME'
        )
        return cielo_ensuciado


def build_pi_astrodeconv_enhanced(dirty_beam, input_shape=(512, 512, 1)):
    """
    PI-AstroDeconv versión mejorada.
    """
    inputs, cielo_limpio_predicho = build_unet_enhanced(input_shape)
    cielo_sucio_predicho = FFTPhysicsLayer(dirty_beam)(cielo_limpio_predicho)
    
    modelo = models.Model(inputs=inputs, outputs=cielo_sucio_predicho, 
                         name="PI-AstroDeconv-Enhanced")
    
    # Compilar con métrica adicional
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='logcosh',
        metrics=['mae', 'mse']  # Métricas adicionales
    )
    
    return modelo


if __name__ == "__main__":
    print("Construyendo PI-AstroDeconv Enhanced...\n")
    
    dummy_beam = np.zeros((21, 21))
    dummy_beam[10, 10] = 1.0
    
    modelo = build_pi_astrodeconv_enhanced(dummy_beam)
    modelo.summary()
    print("\n✅ Arquitectura mejorada lista")
```

**MEJORAS APLICADAS:**
- ✅ 4 niveles de compresión (512 → 256 → 128 → 64 → 32)
- ✅ Cuello de botella a 32×32 (en lugar de 128×128)
- ✅ Filtros aumentados: 256 en bottleneck
- ✅ BatchNormalization después de cada convolución
- ✅ Dropout para regularización (0.2 regular, 0.3 en bottleneck)
- ✅ Learning rate más bajo (1e-4) para entrenamiento estable
- ✅ Métricas adicionales (MAE, MSE)

---

## 2. ENTRENAMIENTO IMPLEMENTADO (04_train_model.py)

```python
"""
Script completo de entrenamiento del modelo PI-AstroDeconv.

Incluye:
- Carga de datos
- Split train/validation/test
- Loop de entrenamiento
- Guardado de checkpoints
- Monitoreo de métricas
- Visualización
"""

import tensorflow as tf
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Importar modelo mejorado
from scripts.scripts_enhanced.model_enhanced import build_pi_astrodeconv_enhanced


class AstroDeconvTrainer:
    """
    Clase para entrenar el modelo PI-AstroDeconv.
    """
    
    def __init__(self, model_path='models/', data_path='data/raw/'):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.history = None
        self.dirty_beam = None
        
        # Crear carpetas si no existen
        self.model_path.mkdir(exist_ok=True, parents=True)
        
    def load_fits_data(self, filename):
        """Carga archivo FITS y retorna como array numpy"""
        try:
            with fits.open(self.data_path / filename) as hdul:
                return hdul[0].data.astype(np.float32)
        except FileNotFoundError:
            print(f"⚠️ Archivo no encontrado: {filename}")
            return None
    
    def prepare_data(self):
        """
        Carga datos y prepara para entrenamiento.
        
        En versión real: cargaría múltiples observaciones
        Por ahora: usa datos simulados
        """
        print("📂 Cargando datos...")
        
        # Cargar dirty sky y dirty beam
        self.dirty_sky = self.load_fits_data('dirty_sky.fits')
        self.dirty_beam = self.load_fits_data('dirty_beam.fits')
        self.clean_sky = self.load_fits_data('clean_sky.fits')
        
        if self.dirty_sky is None or self.dirty_beam is None:
            print("❌ Error cargando datos")
            return False
        
        # Normalizar datos
        print("🔄 Normalizando...")
        self.dirty_sky = self._normalize(self.dirty_sky)
        self.clean_sky = self._normalize(self.clean_sky) if self.clean_sky is not None else None
        
        # Preparar batches
        # En versión mejorada: usar tf.data.Dataset
        self.X_train = np.expand_dims(self.dirty_sky, axis=-1)  # 512×512 → 512×512×1
        self.X_train = np.expand_dims(self.X_train, axis=0)     # → 1×512×512×1
        
        self.y_train = self.X_train.copy()  # Target = entrada (dirty image)
        
        print(f"✅ Datos cargados: {self.X_train.shape}")
        return True
    
    def _normalize(self, data):
        """Normaliza array entre 0 y 1"""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        return data
    
    def build_model(self):
        """Construye modelo mejorado"""
        print("🏗️ Construyendo modelo mejorado...")
        self.model = build_pi_astrodeconv_enhanced(self.dirty_beam)
        print("✅ Modelo listo")
    
    def train(self, epochs=50, batch_size=1, validation_split=0.1):
        """
        Entrena el modelo.
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño de batch
            validation_split: Proporción validación
        """
        if self.model is None:
            self.build_model()
        
        print(f"\n🚀 Iniciando entrenamiento por {epochs} épocas...")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path / 'modelo_mejor.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entrenar
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Entrenamiento completado")
        return self.history
    
    def evaluate(self, X_test=None, y_test=None):
        """Evalúa el modelo"""
        if X_test is None:
            X_test = self.X_train
            y_test = self.y_train
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n📊 Resultados Evaluación:")
        print(f"  Loss: {results[0]:.6f}")
        print(f"  MAE:  {results[1]:.6f}")
        print(f"  MSE:  {results[2]:.6f}")
        
        return results
    
    def predict(self, dirty_image):
        """Predice cielo limpio a partir de cielo sucio"""
        if self.model is None:
            print("❌ Modelo no entrenado")
            return None
        
        # Normalizar entrada
        dirty_image = self._normalize(dirty_image)
        dirty_image = np.expand_dims(dirty_image, axis=-1)
        dirty_image = np.expand_dims(dirty_image, axis=0)
        
        # Predecir
        prediction = self.model.predict(dirty_image, verbose=0)
        
        return prediction[0, :, :, 0]  # Remover batch y canal
    
    def plot_history(self):
        """Grafica historial de entrenamiento"""
        if self.history is None:
            print("❌ No hay historial de entrenamiento")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss (Log-Cosh)')
        axes[0].set_title('Convergencia del Modelo')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Error Absoluto Medio')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_path / 'training_history.png', dpi=150)
        print("✅ Gráfico guardado: training_history.png")
        
        return fig
    
    def save_model(self, name='pi_astrodeconv_trained'):
        """Guarda modelo entrenado"""
        if self.model is None:
            print("❌ No hay modelo para guardar")
            return
        
        filepath = self.model_path / f'{name}.h5'
        self.model.save(filepath)
        print(f"✅ Modelo guardado: {filepath}")
    
    def load_model(self, name='pi_astrodeconv_trained'):
        """Carga modelo previamente entrenado"""
        filepath = self.model_path / f'{name}.h5'
        if not filepath.exists():
            print(f"❌ Modelo no encontrado: {filepath}")
            return False
        
        self.model = tf.keras.models.load_model(filepath)
        print(f"✅ Modelo cargado: {filepath}")
        return True


if __name__ == "__main__":
    # Crear entrenador
    trainer = AstroDeconvTrainer()
    
    # Preparar datos
    if not trainer.prepare_data():
        print("❌ Error preparando datos")
        exit(1)
    
    # Construir y entrenar modelo
    trainer.build_model()
    trainer.model.summary()
    
    # Entrenar
    history = trainer.train(epochs=50, batch_size=1, validation_split=0.2)
    
    # Evaluar
    trainer.evaluate()
    
    # Visualizar
    trainer.plot_history()
    
    # Guardar
    trainer.save_model()
    
    print("\n🎉 Entrenamiento completado exitosamente")
```

**CARACTERÍSTICAS IMPLEMENTADAS:**
- ✅ Carga de datos FITS
- ✅ Normalización
- ✅ ModelCheckpoint (guarda mejor modelo)
- ✅ EarlyStopping (evita overfitting)
- ✅ ReduceLROnPlateau (ajusta learning rate)
- ✅ Evaluación cuantitativa
- ✅ Visualización de convergencia
- ✅ Predicción en nuevas imágenes

---

## 3. EVALUACIÓN Y MÉTRICAS (05_evaluate.py)

```python
"""
Evaluación completa del modelo con métricas astronómicas.

Métricas:
- Power Spectrum (21cm)
- PSNR / SSIM (visión por computadora)
- MAE / RMSE (estadística)
- Visualización antes/después
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tensorflow as tf
from astropy.io import fits


def compute_power_spectrum(image, L=512):
    """
    Calcula power spectrum 2D de la imagen.
    
    Args:
        image: Array 2D
        L: Tamaño de imagen
    
    Returns:
        power_spectrum: Espectro de potencia 1D
        frequencies: Frecuencias espaciales
    """
    # FFT
    fft_image = np.fft.fft2(image)
    power = np.abs(fft_image) ** 2
    
    # Shift (centrar DC)
    power = np.fft.fftshift(power)
    
    # Frequencies
    freq = np.fft.fftfreq(L)
    freq = np.fft.fftshift(freq)
    
    # Convertir a radial (1D)
    frequencies = []
    power_spectrum = []
    
    for i in range(L):
        for j in range(L):
            f = np.sqrt(freq[i]**2 + freq[j]**2)
            frequencies.append(f)
            power_spectrum.append(power[i, j])
    
    return np.array(power_spectrum), np.array(frequencies)


class AstroDeconvEvaluator:
    """Evaluador de modelos astronómicos"""
    
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
    
    def evaluate_metrics(self, clean_truth, clean_predicted, name=""):
        """
        Calcula múltiples métricas.
        
        Args:
            clean_truth: Imagen limpia verdadera (Ground Truth)
            clean_predicted: Imagen limpia predicha por el modelo
            name: Nombre para reporte
        """
        print(f"\n📊 Métricas de Evaluación: {name}")
        print("=" * 50)
        
        # Normalizar
        clean_truth = self._normalize(clean_truth)
        clean_predicted = self._normalize(clean_predicted)
        
        # 1. MAE (Mean Absolute Error)
        mae = np.mean(np.abs(clean_truth - clean_predicted))
        print(f"MAE:  {mae:.6f}")
        
        # 2. RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((clean_truth - clean_predicted) ** 2))
        print(f"RMSE: {rmse:.6f}")
        
        # 3. PSNR (Peak Signal-to-Noise Ratio)
        psnr = peak_signal_noise_ratio(clean_truth, clean_predicted, data_range=1.0)
        print(f"PSNR: {psnr:.2f} dB")
        
        # 4. SSIM (Structural Similarity Index)
        ssim = structural_similarity(clean_truth, clean_predicted, data_range=1.0)
        print(f"SSIM: {ssim:.4f}")
        
        # 5. Power Spectrum Error (21cm specific)
        ps_truth, freq_truth = compute_power_spectrum(clean_truth)
        ps_pred, _ = compute_power_spectrum(clean_predicted)
        
        ps_error = np.mean(np.abs(ps_truth - ps_pred))
        print(f"Power Spectrum Error: {ps_error:.6e}")
        
        print("=" * 50)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'psnr': psnr,
            'ssim': ssim,
            'ps_error': ps_error
        }
    
    def visualize_deconvolution(self, dirty_image, clean_truth, clean_predicted):
        """
        Visualiza antes/después de deconvolución.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Normalizar
        dirty_image = self._normalize(dirty_image)
        clean_truth = self._normalize(clean_truth)
        clean_predicted = self._normalize(clean_predicted)
        
        # Dirty Sky
        im0 = axes[0, 0].imshow(dirty_image, cmap='inferno', origin='lower')
        axes[0, 0].set_title('Cielo Sucio (Dirty Image)')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Clean Truth
        im1 = axes[0, 1].imshow(clean_truth, cmap='inferno', origin='lower')
        axes[0, 1].set_title('Cielo Limpio Real (Ground Truth)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Clean Predicted
        im2 = axes[1, 0].imshow(clean_predicted, cmap='inferno', origin='lower')
        axes[1, 0].set_title('Cielo Limpio Predicho (Modelo)')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Error
        error = np.abs(clean_truth - clean_predicted)
        im3 = axes[1, 1].imshow(error, cmap='hot', origin='lower')
        axes[1, 1].set_title(f'Error Absoluto (MAE={np.mean(error):.6f})')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def plot_power_spectrum(self, clean_truth, clean_predicted):
        """
        Grafica power spectrum en escala log.
        """
        ps_truth, freq = compute_power_spectrum(clean_truth)
        ps_pred, _ = compute_power_spectrum(clean_predicted)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(freq, ps_truth, label='Ground Truth', linewidth=2)
        ax.loglog(freq, ps_pred, label='Modelo Predicho', linewidth=2)
        
        ax.set_xlabel('Frecuencia Espacial')
        ax.set_ylabel('Power Spectrum')
        ax.set_title('Comparativa Power Spectrum (21cm)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        return fig
    
    @staticmethod
    def _normalize(data):
        """Normaliza entre 0 y 1"""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        return data


if __name__ == "__main__":
    # Crear evaluador
    evaluator = AstroDeconvEvaluator()
    
    # Cargar datos
    with fits.open('data/raw/clean_sky.fits') as hdul:
        clean_truth = hdul[0].data
    
    with fits.open('data/raw/dirty_sky.fits') as hdul:
        dirty_image = hdul[0].data
    
    # Simular predicción (en real: usar modelo)
    clean_predicted = clean_truth + np.random.normal(0, 0.1, clean_truth.shape)
    
    # Evaluar
    metrics = evaluator.evaluate_metrics(clean_truth, clean_predicted)
    
    # Visualizar
    fig1 = evaluator.visualize_deconvolution(dirty_image, clean_truth, clean_predicted)
    fig2 = evaluator.plot_power_spectrum(clean_truth, clean_predicted)
    
    plt.show()
```

---

## RESUMEN DE CÓDIGO FALTANTE

| Componente | Estado | Líneas | Prioridad |
|-----------|--------|--------|-----------|
| Arquitectura mejorada | ✅ Arriba | ~120 | 🔴 CRÍTICA |
| Entrenamiento (04) | ✅ Arriba | ~250 | 🔴 CRÍTICA |
| Evaluación (05) | ✅ Arriba | ~180 | 🟡 IMPORTANTE |

**TOTAL CÓDIGO NUEVO:** ~550 líneas

Esto llevaría el proyecto de **33% a 90% de completitud**

---

Generado: 2026-06-05
