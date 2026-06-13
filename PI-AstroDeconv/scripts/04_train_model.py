"""
PI-AstroDeconv: Script de Entrenamiento

Entrena el modelo PI-AstroDeconv mejorado con:
- Carga de datos FITS
- Normalización
- Callbacks (checkpoints, early stopping, learning rate reduction)
- Monitoreo de convergencia
- Visualización de resultados
"""

import tensorflow as tf
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import importlib.util

# Importar el modelo mejorado (desde 03_model_unet.py)
spec = importlib.util.spec_from_file_location("model", Path(__file__).parent / "03_model_unet.py")
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
build_pi_astrodeconv = model_module.build_pi_astrodeconv


class AstroDeconvTrainer:
    """
    Clase para entrenar el modelo PI-AstroDeconv de forma semi-supervisada.
    
    La loss se calcula en el espacio ensuciado (dirty space):
    Loss = MSE(Dirty_Sky_Real, Dirty_Sky_Predicho)
    
    Donde:
    - Dirty_Sky_Predicho = FFT(Dirty_Beam ⊗ U-Net(Dirty_Sky_Real))
    - No necesita Ground Truth (aprendizaje no supervisado)
    """
    
    def __init__(self, model_path='models/', data_path='data/raw/'):
        """
        Inicializa el entrenador.
        
        Args:
            model_path: Directorio para guardar checkpoints
            data_path: Directorio con datos FITS
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.history = None
        self.dirty_beam = None
        self.dirty_sky = None
        self.clean_sky = None
        
        # Crear carpeta de modelos si no existe
        self.model_path.mkdir(exist_ok=True, parents=True)
        
    def load_fits_data(self, filename):
        """
        Carga archivo FITS y lo convierte a numpy array.
        
        Args:
            filename: Nombre del archivo FITS
        
        Returns:
            Array numpy con los datos, None si no existe
        """
        filepath = self.data_path / filename
        try:
            with fits.open(filepath) as hdul:
                data = hdul[0].data
                return data.astype(np.float32)
        except FileNotFoundError:
            print(f"⚠️  Archivo no encontrado: {filepath}")
            return None
    
    def prepare_data(self):
        """
        Carga datos FITS y prepara para entrenamiento.
        
        Carga:
        - dirty_sky.fits: Imagen observada
        - dirty_beam.fits: Point Spread Function (PSF)
        - clean_sky.fits: Ground truth (opcional, para validación)
        
        Returns:
            True si fue exitoso, False si hubo error
        """
        print("\n📂 Cargando datos FITS...")
        
        # Cargar datos
        self.dirty_sky = self.load_fits_data('dirty_sky.fits')
        self.dirty_beam = self.load_fits_data('dirty_beam.fits')
        self.clean_sky = self.load_fits_data('clean_sky.fits')
        
        # Verificar que están disponibles
        if self.dirty_sky is None or self.dirty_beam is None:
            print("❌ Error: Falta dirty_sky.fits o dirty_beam.fits")
            return False
        
        print(f"✅ Dirty Sky: {self.dirty_sky.shape}")
        print(f"✅ Dirty Beam: {self.dirty_beam.shape}")
        if self.clean_sky is not None:
            print(f"✅ Clean Sky (Ground Truth): {self.clean_sky.shape}")
        
        # Normalizar datos
        print("\n🔄 Normalizando...")
        self.dirty_sky = self._normalize(self.dirty_sky)
        self.dirty_beam = self._normalize(self.dirty_beam)
        if self.clean_sky is not None:
            self.clean_sky = self._normalize(self.clean_sky)
        
        # Preparar batches para entrenamiento
        # Entrada: imagen sucia (512×512) → (1, 512, 512, 1)
        self.X_train = np.expand_dims(self.dirty_sky, axis=-1)  # Add channel
        self.X_train = np.expand_dims(self.X_train, axis=0)     # Add batch
        
        # Target: imagen sucia (no cambia, es semi-supervisado)
        self.y_train = self.X_train.copy()
        
        print(f"\n✅ Datos preparados:")
        print(f"   X_train shape: {self.X_train.shape}")
        print(f"   y_train shape: {self.y_train.shape}")
        
        return True
    
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
    
    def build_model(self):
        """
        Construye el modelo mejorado.
        """
        print("\n🏗️ Construyendo modelo mejorado...")
        h, w = self.dirty_sky.shape[:2]
        input_shape = (h, w, 1)
        self.model = build_pi_astrodeconv(self.dirty_beam, input_shape=input_shape)
        print(f"✅ Modelo construido (input_shape={input_shape})")
    
    def train(self, epochs=50, batch_size=1, validation_split=0):
        """
        Entrena el modelo con callbacks.
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño de batch (usualmente 1 para imágenes grandes)
            validation_split: Proporción de validación (0 = sin validación)
        
        Returns:
            Historia de entrenamiento (objeto History de Keras)
        """
        if self.model is None:
            self.build_model()
        
        print(f"\n🚀 Iniciando entrenamiento por {epochs} épocas...")
        print(f"   Learning rate inicial: 1e-4")
        print(f"   Loss: Log-Cosh (robusto a outliers)")
        print(f"   Batch size: {batch_size}")
        
        # Definir callbacks
        callbacks = [
            # Guardar mejor modelo (monitorear loss de entrenamiento)
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path / 'modelo_mejor.h5'),
                monitor='loss' if validation_split == 0 else 'val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            ),
            # Detener si no mejora
            tf.keras.callbacks.EarlyStopping(
                monitor='loss' if validation_split == 0 else 'val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True,
                mode='min'
            ),
            # Reducir learning rate si se estanca
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss' if validation_split == 0 else 'val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            # Ver progreso en tiempo real
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.model_path / 'logs'),
                histogram_freq=1
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
        
        print("\n✅ Entrenamiento completado")
        return self.history
    
    def evaluate(self, X_test=None, y_test=None):
        """
        Evalúa el modelo en datos de prueba.
        
        Args:
            X_test: Imágenes de prueba (si None, usa train)
            y_test: Targets de prueba (si None, usa train)
        
        Returns:
            Tupla (loss, mae, mse)
        """
        if X_test is None:
            X_test = self.X_train
            y_test = self.y_train
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n📊 Resultados Evaluación:")
        print(f"   Loss (Log-Cosh): {results[0]:.6f}")
        print(f"   MAE:             {results[1]:.6f}")
        print(f"   MSE:             {results[2]:.6f}")
        
        return results
    
    def predict(self, dirty_image):
        """
        Predice cielo limpio a partir de cielo sucio.
        
        Args:
            dirty_image: Array 2D con imagen sucia
        
        Returns:
            Array 2D con predicción de cielo limpio
        """
        if self.model is None:
            print("❌ Error: Modelo no entrenado")
            return None
        
        # Normalizar
        dirty_image = self._normalize(dirty_image)
        
        # Agregar dimensiones: 512×512 → 1×512×512×1
        dirty_image_batch = np.expand_dims(dirty_image, axis=-1)
        dirty_image_batch = np.expand_dims(dirty_image_batch, axis=0)
        
        # Predecir
        prediction = self.model.predict(dirty_image_batch, verbose=0)
        
        # Remover batch y canal: 1×512×512×1 → 512×512
        return prediction[0, :, :, 0]
    
    def plot_history(self):
        """
        Grafica el historial de entrenamiento.
        
        Muestra:
        - Loss de entrenamiento y validación
        - MAE de entrenamiento y validación
        """
        if self.history is None:
            print("❌ No hay historial de entrenamiento")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        has_val = 'val_loss' in self.history.history

        # Gráfico 1: Loss
        axes[0].plot(self.history.history['loss'],
                    label='Train Loss', linewidth=2)
        if has_val:
            axes[0].plot(self.history.history['val_loss'],
                        label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Loss (Log-Cosh)', fontsize=12)
        axes[0].set_title('Convergencia del Modelo', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Gráfico 2: MAE
        axes[1].plot(self.history.history['mae'],
                    label='Train MAE', linewidth=2)
        if has_val:
            axes[1].plot(self.history.history['val_mae'],
                        label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Error Absoluto Medio', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        output_path = self.model_path / 'training_history.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Gráfico guardado: {output_path}")
        
        return fig
    
    def save_model(self, name='pi_astrodeconv_trained'):
        """
        Guarda el modelo entrenado.
        
        Args:
            name: Nombre del archivo (sin .h5)
        """
        if self.model is None:
            print("❌ Error: No hay modelo para guardar")
            return
        
        filepath = self.model_path / f'{name}.h5'
        self.model.save(filepath)
        print(f"\n✅ Modelo guardado: {filepath}")
    
    def load_model(self, name='pi_astrodeconv_trained'):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            name: Nombre del archivo guardado (sin .h5)
        
        Returns:
            True si fue exitoso, False si no
        """
        filepath = self.model_path / f'{name}.h5'
        if not filepath.exists():
            print(f"❌ Error: Modelo no encontrado: {filepath}")
            return False
        
        self.model = tf.keras.models.load_model(filepath)
        print(f"✅ Modelo cargado: {filepath}")
        return True


def main():
    """
    Pipeline completo de entrenamiento.
    """
    print("\n" + "="*60)
    print("PI-AstroDeconv: Pipeline de Entrenamiento")
    print("="*60)
    
    # Crear entrenador
    trainer = AstroDeconvTrainer(
        model_path='models/',
        data_path='data/raw/'
    )
    
    # Preparar datos
    if not trainer.prepare_data():
        print("❌ No se pudieron preparar los datos")
        return False
    
    # Construir modelo
    trainer.build_model()
    trainer.model.summary()
    
    # Entrenar
    print("\n" + "-"*60)
    print("FASE 1: ENTRENAMIENTO")
    print("-"*60)
    history = trainer.train(epochs=50, batch_size=1, validation_split=0)
    
    # Evaluar
    print("\n" + "-"*60)
    print("FASE 2: EVALUACIÓN")
    print("-"*60)
    trainer.evaluate()
    
    # Visualizar convergencia
    print("\n" + "-"*60)
    print("FASE 3: VISUALIZACIÓN")
    print("-"*60)
    trainer.plot_history()
    
    # Guardar modelo
    trainer.save_model('pi_astrodeconv_trained')
    
    print("\n" + "="*60)
    print("✅ Entrenamiento completado exitosamente")
    print("="*60)
    
    return True


if __name__ == "__main__":
    main()
