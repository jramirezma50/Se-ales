"""
03_model_unet.py

Aquí construimos la Inteligencia Artificial (PI-AstroDeconv).
Usaremos TensorFlow para crear la red neuronal U-Net y le
añadiremos nuestra capa especial "Física" al final.
"""

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import numpy as np

def build_unet(input_shape=(512, 512, 1)):
    """
    Construye la arquitectura base de la U-Net.
    (Forma de U: bajada/compresión y subida/reconstrucción)
    """
    inputs = layers.Input(shape=input_shape, name="imagen_sucia_entrada")

    # --- ENCODER (La Bajada) ---
    # Extraemos características y reducimos tamaño
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # --- CUELLO DE BOTELLA ---
    # La parte más profunda de la red, donde "comprende" la imagen
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)

    # --- DECODER (La Subida) ---
    # Agrandamos y usamos la información guardada (Skip connections)
    u1 = layers.UpSampling2D((2, 2))(c3)
    concat1 = layers.Concatenate()([u1, c2]) # Unimos con lo que recordamos del paso c2
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)

    u2 = layers.UpSampling2D((2, 2))(c4)
    concat2 = layers.Concatenate()([u2, c1]) # Unimos con lo que recordamos del paso c1
    c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(concat2)

    # La salida de la U-Net es nuestra predicción del CIELO LIMPIO
    # Usamos activación 'relu' porque en astronomía no existe el "brillo negativo"
    cielo_limpio_predicho = layers.Conv2D(1, (1, 1), activation='relu', name="cielo_limpio_predicho")(c5)

    return inputs, cielo_limpio_predicho

class FFTPhysicsLayer(layers.Layer):
    """
    ¡LA MAGIA DEL PAPER!
    Esta capa no tiene neuronas que aprenden. Toma el "Cielo Limpio" que adivinó 
    la U-Net y lo ensucia matemáticamente usando el Dirty Beam del telescopio.
    """
    def __init__(self, dirty_beam, **kwargs):
        super(FFTPhysicsLayer, self).__init__(**kwargs)
        # Convertimos nuestro dirty beam matemático a un formato que TensorFlow entienda
        # Shape requerido por TensorFlow: (Alto, Ancho, Canales_Entrada, Canales_Salida)
        beam_tensor = tf.convert_to_tensor(dirty_beam, dtype=tf.float32)
        beam_tensor = tf.expand_dims(tf.expand_dims(beam_tensor, -1), -1)
        self.dirty_beam = beam_tensor

    def call(self, inputs):
        # Aplicamos la convolución. 
        # TensorFlow optimiza internamente esto con algoritmos FFT para que sea rapidísimo.
        cielo_ensuciado = tf.nn.conv2d(
            inputs, 
            self.dirty_beam, 
            strides=[1, 1, 1, 1], 
            padding='SAME',
            name="convolucion_fisica"
        )
        return cielo_ensuciado

def build_pi_astrodeconv(dirty_beam, input_shape=(512, 512, 1)):
    """
    Une la U-Net con la Capa Física y prepara todo para entrenar.
    """
    # 1. Creamos la U-Net
    inputs, cielo_limpio_predicho = build_unet(input_shape)
    
    # 2. Le pegamos la capa física al final
    cielo_sucio_predicho = FFTPhysicsLayer(dirty_beam, name="salida_ensuciada")(cielo_limpio_predicho)
    
    # 3. Empaquetamos todo el modelo
    modelo = models.Model(inputs=inputs, outputs=cielo_sucio_predicho, name="PI-AstroDeconv")
    
    # 4. Compilamos el modelo
    # El paper menciona específicamente el optimizador Adam y la pérdida Log-Cosh 
    # por su estabilidad matemática ante valores atípicos (outliers).
    modelo.compile(optimizer='adam', loss='logcosh')
    
    return modelo

if __name__ == "__main__":
    print(" Construyendo la arquitectura PI-AstroDeconv en la memoria RAM...\n")
    
    # Creamos un Beam falso pequeño sólo para probar que el código compila bien
    dummy_beam = np.zeros((21, 21)) 
    dummy_beam[10, 10] = 1.0 
    
    # Ensamblamos la IA
    modelo = build_pi_astrodeconv(dummy_beam)
    
    # Imprimimos el mapa de la arquitectura
    modelo.summary()
    print("\n✅ Arquitectura de Inteligencia Artificial construida exitosamente.")
