"""
PI-AstroDeconv: Modelo U-Net Mejorado con Física

Versión mejorada del modelo U-Net con:
- Mayor profundidad (4-5 niveles de compresión)
- BatchNormalization después de cada convolución
- Dropout para regularización
- Cuello de botella más agresivo
- Capa de física FFT integrada
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from scipy.signal import fftconvolve


def build_unet_enhanced(input_shape=(512, 512, 1)):
    """
    U-Net mejorada con profundidad y regularización.
    
    Arquitectura:
    512 → 256 → 128 → 64 → 32 (cuello) → 64 → 128 → 256 → 512
    
    Cada bloque incluye:
    - Convolución 3x3
    - BatchNormalization
    - ReLU activation
    - Dropout regularization
    """
    inputs = layers.Input(shape=input_shape, name="imagen_sucia")
    
    # ===== ENCODER (Bajada) =====
    # Nivel 1: 512 × 512
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Dropout(0.2)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)  # 512 → 256
    
    # Nivel 2: 256 × 256
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Dropout(0.2)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)  # 256 → 128
    
    # Nivel 3: 128 × 128
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Dropout(0.2)(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)  # 128 → 64
    
    # Nivel 4: 64 × 64
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.2)(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)  # 64 → 32
    
    # ===== CUELLO DE BOTELLA (Bottleneck) =====
    # Punto más comprimido: 32 × 32
    # Aquí es donde la red "entiende" la esencia de la imagen
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.Dropout(0.3)(bottleneck)  # Dropout más alto aquí
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(bottleneck)
    bottleneck = layers.BatchNormalization()(bottleneck)
    
    # ===== DECODER (Subida) =====
    # Nivel 4 (Invertido): 32 → 64
    u4 = layers.UpSampling2D((2, 2))(bottleneck)
    u4 = layers.Concatenate()([u4, c4])  # Skip connection
    d4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.Dropout(0.2)(d4)
    
    # Nivel 3: 64 → 128
    u3 = layers.UpSampling2D((2, 2))(d4)
    u3 = layers.Concatenate()([u3, c3])  # Skip connection
    d3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(0.2)(d3)
    
    # Nivel 2: 128 → 256
    u2 = layers.UpSampling2D((2, 2))(d3)
    u2 = layers.Concatenate()([u2, c2])  # Skip connection
    d2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Dropout(0.2)(d2)
    
    # Nivel 1: 256 → 512
    u1 = layers.UpSampling2D((2, 2))(d2)
    u1 = layers.Concatenate()([u1, c1])  # Skip connection
    d1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Dropout(0.2)(d1)
    
    # ===== SALIDA FINAL =====
    output = layers.Conv2D(1, (1, 1), activation='relu', name="cielo_limpio_predicho")(d1)
    
    return inputs, output


class FFTPhysicsLayer(layers.Layer):
    """
    Capa personalizada que aplica física mediante convolución FFT.
    
    Simula: Cielo_Sucio_Predicho = Dirty_Beam ⊗ Cielo_Limpio_Predicho
    
    Donde ⊗ es convolución, implementada con FFT para eficiencia.
    """
    
    def __init__(self, dirty_beam, **kwargs):
        super(FFTPhysicsLayer, self).__init__(**kwargs)
        # Convertir dirty_beam a tensor de TensorFlow
        beam_tensor = tf.convert_to_tensor(dirty_beam, dtype=tf.float32)
        beam_tensor = tf.expand_dims(tf.expand_dims(beam_tensor, -1), -1)
        self.dirty_beam = beam_tensor

    def call(self, inputs):
        """
        Aplica convolución con el dirty beam.
        
        TensorFlow optimiza esto internamente con FFT.
        """
        cielo_ensuciado_predicho = tf.nn.conv2d(
            inputs, 
            self.dirty_beam, 
            strides=[1, 1, 1, 1], 
            padding='SAME'
        )
        return cielo_ensuciado_predicho


def build_pi_astrodeconv_enhanced(dirty_beam, input_shape=(512, 512, 1)):
    """
    Construye el modelo completo PI-AstroDeconv mejorado.
    
    Arquitectura:
    Entrada → U-Net Mejorada → Predicción Cielo Limpio → Capa Física → Salida
    
    Args:
        dirty_beam: Array numpy con el Dirty Beam (PSF)
        input_shape: Forma de entrada (alto, ancho, canales)
    
    Returns:
        model: Modelo de Keras compilado
    """
    # Construir U-Net
    inputs, cielo_limpio_predicho = build_unet_enhanced(input_shape)
    
    # Aplicar capa de física
    cielo_sucio_predicho = FFTPhysicsLayer(dirty_beam)(cielo_limpio_predicho)
    
    # Crear modelo
    modelo = models.Model(
        inputs=inputs, 
        outputs=cielo_sucio_predicho, 
        name="PI-AstroDeconv-Enhanced"
    )
    
    # Compilar con optimizador mejorado
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='logcosh',  # Robusto a outliers
        metrics=['mae', 'mse']
    )
    
    return modelo


if __name__ == "__main__":
    print("=" * 60)
    print("PI-AstroDeconv: Arquitectura Mejorada")
    print("=" * 60)
    
    # Crear dirty beam dummy (para prueba)
    dummy_beam = np.zeros((21, 21))
    dummy_beam[10, 10] = 1.0
    
    # Construir modelo
    print("\n🏗️ Construyendo modelo mejorado...")
    modelo = build_pi_astrodeconv_enhanced(dummy_beam)
    
    # Mostrar resumen
    print("\n📊 Resumen de la arquitectura:")
    modelo.summary()
    
    # Contar parámetros
    total_params = modelo.count_params()
    print(f"\n✅ Total de parámetros: {total_params:,}")
    print("✅ Arquitectura mejorada lista para entrenar")
    print("=" * 60)
