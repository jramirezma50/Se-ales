# Guía Maestra de Exposición: PI-AstroDeconv (Fase de Simulación de Datos)

Este documento contiene toda la información necesaria para exponer y defender el trabajo realizado hasta el momento. Está diseñado para que domines la física, las matemáticas y el código detrás de la generación de datos para el proyecto PI-AstroDeconv.

---

## 1. Contexto Físico y Astronómico

### El Objetivo
Queremos mapear el universo temprano observando el **Hidrógeno Neutro (HI)**. Este gas emite una débil señal de radio a una longitud de onda de **21 cm**. Para captarla, proyectos como el SKA (Square Kilometre Array) usan cientos de antenas conectadas (Interferometría).

                ### El Problema
Como las antenas no cubren cada milímetro del suelo, tenemos "agujeros" en nuestra visión. Esto genera un defecto óptico matemático llamado **Point Spread Function (PSF)** o **Dirty Beam**. Esto significa que si el telescopio mira una estrella perfecta (un punto de luz), no verá un punto, sino una mancha central rodeada de anillos de difracción. Por lo tanto, el cielo que vemos a través del telescopio es un "Cielo Sucio" (Dirty Sky).

---

## 2. ¿Qué son los archivos FITS?
FITS (*Flexible Image Transport System*) es el formato estándar mundial en astronomía. 
A diferencia de un JPG o PNG que guarda "píxeles de colores", un FITS guarda **matrices matemáticas puras** (intensidad de radiación, flujo de fotones, etc.) con una precisión flotante inmensa. 
Además, contiene un **Header (Cabecera)**, que es un diccionario de metadatos que indica en qué coordenadas exactas del cielo se tomó la imagen, con qué telescopio y en qué fecha.

---

## 3. Análisis del Código 1: `01_check_fits.py`

Este script fue nuestro "Hola Mundo" en la astronomía. Su objetivo es enseñarnos cómo abrir e inspeccionar un archivo FITS.

### Librerías Utilizadas:
- `import os`: Interactúa con el sistema operativo (para verificar si el archivo existe en el disco duro).
- `import numpy as np`: La librería suprema de matemáticas en Python. Maneja matrices gigantes y vectores a alta velocidad.
- `import matplotlib.pyplot as plt`: El motor de gráficos para dibujar las imágenes.
- `from astropy.io import fits`: La herramienta específica que sabe cómo "descomprimir" la estructura interna de un FITS.

### Desglose línea por línea de la lógica:
```python
def inspect_fits(file_path):
```
Creamos una función que recibe la ruta de un archivo.

```python
    if not os.path.exists(file_path):
        datos_simulados = np.random.normal(0, 1, (512, 512))
```
Si el archivo FITS no existe (porque no lo habíamos descargado/creado aún), usamos `numpy` para crear una matriz falsa de **512x512**. `np.random.normal(0, 1)` genera números aleatorios simulando el ruido blanco del espacio electromagnético.

```python
    with fits.open(file_path) as hdul:
```
Si el archivo sí existe, usamos `fits.open`. `hdul` significa *Header Data Unit List*. Un FITS es como una cebolla con varias capas. Abrimos el archivo temporalmente.

```python
        hdul.info()
        datos = hdul[0].data
        cabecera = hdul[0].header 
```
`.info()` imprime qué capas tiene el archivo.
`hdul[0].data` extrae la matriz matemática pura de la primera capa (PrimaryHDU).
`hdul[0].header` extrae los metadatos astronómicos.

---

## 4. Análisis del Código 2: `02_simulate_data.py`

Dado que generar datos con las simuladoras profesionales (OSKAR/21cmFAST) requiere supercomputadoras, este script **fabrica un universo matemático de bolsillo** para poder entrenar a nuestra Inteligencia Artificial en nuestra propia computadora.

### A) Función `create_clean_sky(size=512)`
**Propósito:** Crear la "Verdad absoluta". Un cielo perfecto sin errores de telescopio.
- `sky = np.zeros((size, size))`: Crea un lienzo negro (matriz llena de ceros).
- El bucle `for _ in range(20):` itera 20 veces para crear 20 fuentes (galaxias).
- `x` y `y` se eligen al azar con `np.random.randint()`.
- `sky[x, y] += brightness`: Le inyectamos valor (brillo) a ese punto del espacio. Sumamos a los píxeles adyacentes para darle un suavizado básico.

### B) Función `create_dirty_beam(size=512)`
**Propósito:** Simular el defecto físico (interferencia óptica) de nuestro arreglo de antenas.
- `x, y = np.linspace(-10, 10, size)`: Crea un plano de coordenadas de -10 a 10 dividido en 512 pasos.
- `X, Y = np.meshgrid(x, y)`: Crea la cuadrícula 2D.
- `R = np.sqrt(X**2 + Y**2)`: Teorema de Pitágoras para calcular el radio (distancia desde el centro para cada píxel).
- **La Ecuación del Defecto**: `beam = np.sin(R * 4) / (R * 4) * np.exp(-R**2 / 10)`
  - Esto es una función `Sinc` (Seno Cardinal) multiplicada por una caída Gaussiana. 
  - La función matemática `Sinc` es famosa en la física de ondas porque representa exactamente el patrón de difracción de la luz/radio al pasar por aperturas redondas (o antenas). Genera ese núcleo brillante central y las "ondas" expansivas.
- `beam /= np.sum(beam)`: Se divide por la suma total para que el "peso" total de la mancha sea 1 (Normalización de energía).

### C) Función `main()` y Convolución FFT
- `from scipy.signal import fftconvolve`: Importamos el algoritmo de la Transformada Rápida de Fourier.
- `dirty_sky = fftconvolve(clean_sky, dirty_beam, mode='same')`
  - **Física de esto:** La imagen sucia es matemáticamente el cielo perfecto *convolucionado* con la mancha del telescopio. En el espacio normal, multiplicar píxel por píxel tomaría horas. La FFT convierte las matrices a frecuencias, las multiplica instantáneamente y las devuelve al mundo espacial. Es la misma matemática que usaremos en la IA de PI-AstroDeconv.
- `save_to_fits`: Usa `fits.PrimaryHDU(data)` para convertir nuestra matriz en un archivo FITS estándar y lo guarda en `/data/raw`.

---

## 5. Explicación de la Imagen Generada (`comparacion_simulacion.png`)

La imagen final consta de 3 paneles y es fundamental para la exposición. 
*¿Qué significan las escalas de colores (Colorbars)?*
La barra al lado de cada imagen indica la **intensidad de la señal (flujo/brillo)**. En astronomía de radio, no hay colores reales, solo intensidades de energía.

1. **Panel 1: Cielo Limpio (La Verdad)**
   - **Escala de Color (`inferno`)**: El fondo es negro (0 energía). Los puntos amarillo/blanco representan alta energía (nuestras galaxias).
   - **Descripción**: Muestra la matriz original donde los puntos son nítidos y finos.

2. **Panel 2: Dirty Beam (El Defecto)**
   - **Escala de Color (`viridis`)**: Se usó otra paleta para resaltar mejor las ondas. El amarillo brillante central es la mayor concentración del error, mientras que las ondas cian/verde son las oscilaciones de difracción.
   - **Construcción**: Hemos hecho un "zoom" a los píxeles centrales de nuestra matriz de 512x512 para que se aprecie la función Sinc. Este patrón es el "sello de agua" destructivo que el telescopio le pondrá a todo lo que vea.

3. **Panel 3: Dirty Sky (La Observación)**
   - **Escala de Color (`inferno`)**: Similar al cielo limpio, pero si te fijas en los valores de la escala y en la textura, notarás el desastre.
   - **Construcción**: Es la combinación geométrica de los dos paneles anteriores.
   - **Explicación para el público**: *"Observen cómo los puntos perfectamente nítidos del Cielo Limpio ahora están envueltos en halos y manchas. Peor aún, las manchas de galaxias cercanas se solapan entre sí (interferencia constructiva/destructiva), creando falsas estructuras y subiendo el brillo del fondo oscuro. Si le entregáramos esto a un científico, pensaría que hay gas donde no lo hay. El objetivo de nuestra futura Inteligencia Artificial es recibir la imagen 3 (Dirty Sky), y usando el patrón de la imagen 2 (Dirty Beam), aprender a devolvernos matemáticamente la imagen 1 (Cielo Limpio)."*

---

## 6. Avance del Proyecto: Implementación de la Inteligencia Artificial (`03_model_unet.py`)

Luego de haber dominado la simulación y física de los datos astronómicos, el proyecto avanzó hacia la construcción de su núcleo computacional: la Inteligencia Artificial propuesta en el paper.

A diferencia de las IAs convencionales a las que se les provee la "respuesta correcta" para que aprendan, aquí en astronomía no tenemos imágenes perfectas y limpias del universo real para enseñarle a la red. Por eso, el paper propone una arquitectura de IA **No Supervisada Informada por Física**. 
Esto significa que la red neuronal debe adivinar un "Cielo Limpio" a ciegas, y ella misma se "ensucia" utilizando una capa especial que aplica las ecuaciones matemáticas del telescopio. Si su cielo sucio inventado coincide con el cielo sucio real observado, la red sabe que su deducción del cielo limpio era la correcta.

A continuación, el análisis detallado línea por línea de cómo se implementó esta arquitectura:

### Librerías Utilizadas
- `import tensorflow as tf`: El framework principal de Inteligencia Artificial de Google, especializado en redes neuronales profundas y operaciones matemáticas con tensores (matrices hiperdimensionales) a través de tarjetas gráficas.
- `from tensorflow.keras import layers, models`: Keras es la interfaz que permite ensamblar la arquitectura de la red neuronal de forma modular y estructurada.

### Desglose línea por línea: La Arquitectura U-Net
```python
def build_unet(input_shape=(512, 512, 1)):
```
Se define la función constructora de la IA, preparada para ingerir nuestras matrices de 512x512 con 1 solo canal de color (la intensidad de radio).

```python
    inputs = layers.Input(shape=input_shape, name="imagen_sucia_entrada")
```
La puerta de entrada. Por aquí alimentaremos a la IA con el mapa del "Cielo Sucio".

**EL ENCODER (Fase de Compresión)**
```python
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
```
- `Conv2D`: La primera capa de convolución neuronal. Aplica 16 "filtros" de 3x3 píxeles buscando extraer características visuales (patrones de brillo, bordes de los halos). La función de activación `relu` se asegura de anular cualquier valor negativo que intente predecir la red (en física, no existe la emisión de energía negativa).
- `MaxPooling2D`: Comprime la resolución de la imagen a la mitad, forzando a la IA a condensar la información y prescindir del ruido innecesario.
*(Este mismo concepto se repite para generar la capa `c2` y `p2`, comprimiéndola aún más y subiendo a 32 filtros).*

**EL CUELLO DE BOTELLA**
```python
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
```
Es el punto más profundo de la red, donde la compresión espacial es máxima. Aquí la IA ya no "ve" píxeles, sino que "comprende" matemáticamente la estructura general y la distribución de las galaxias.

**EL DECODER (Fase de Reconstrucción y Skip Connections)**
```python
    u1 = layers.UpSampling2D((2, 2))(c3)
    concat1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)
```
- `UpSampling2D`: Ahora vamos de subida. Expandimos matemáticamente la imagen para recuperar resolución.
- `Concatenate`: **La genialidad histórica del diseño U-Net.** Esta función implementa lo que se llama una *Skip Connection* (Conexión Residual). Aquí la red fusiona su "comprensión general" (`u1`) con los recuerdos geométricos exactos que guardó antes de comprimir (`c2`). Esto le permite pintar la imagen final con una precisión de píxel casi perfecta.
*(Este proceso se repite para regresar la imagen a su tamaño masivo y original de 512x512 en la capa `c5`).*

```python
    cielo_limpio_predicho = layers.Conv2D(1, (1, 1), activation='relu')(c5)
```
La última capa de la U-Net colapsa toda la información procesada en una única capa usando un filtro de convolución de 1x1. El resultado final de esta operación es la "adivinanza" de la red: el **Cielo Limpio Predicho**.

---

### Desglose línea por línea: La Capa de Convolución Física (`FFTPhysicsLayer`)
Para implementar la parte de "Física" del paper PI-AstroDeconv, construimos una capa que **no aprende nada** (no tiene neuronas), su único trabajo es ensuciar.

```python
class FFTPhysicsLayer(layers.Layer):
    def __init__(self, dirty_beam, **kwargs):
        super(FFTPhysicsLayer, self).__init__(**kwargs)
```
Creamos una clase personalizada. Al momento de invocarla, le inyectamos la matriz de nuestro defecto de telescopio (*Dirty Beam*).

```python
        beam_tensor = tf.convert_to_tensor(dirty_beam, dtype=tf.float32)
        beam_tensor = tf.expand_dims(tf.expand_dims(beam_tensor, -1), -1)
        self.dirty_beam = beam_tensor
```
Convertimos la matriz del Dirty Beam en un **Tensor** (el formato nativo hiperdimensional de TensorFlow). La función `expand_dims` añade dimensiones invisibles (Canal de Entrada y Canal de Salida) para que la forma matemática cuadre con los estándares de convolución de la arquitectura GPU.

```python
    def call(self, inputs):
        cielo_ensuciado = tf.nn.conv2d(inputs, self.dirty_beam, strides=[1,1,1,1], padding='SAME')
        return cielo_ensuciado
```
El método `call` se dispara cuando las imágenes atraviesan esta capa. Toma el cielo limpio adivinado por la U-Net (`inputs`) y le estrella geométricamente el Dirty Beam encima usando `tf.nn.conv2d`. 
**Nota del Paper:** Detrás de cámaras, TensorFlow acelera esta función específica (`conv2d`) implementando algoritmos de Transformada Rápida de Fourier (FFT). Esto cumple el hito técnico descrito en la literatura de PI-AstroDeconv, permitiendo que la red se entrene en tiempos prácticos y no tarde decenas de años.

---

### Ensamblando el Cerebro Final: `build_pi_astrodeconv`
```python
def build_pi_astrodeconv(dirty_beam, input_shape=(512, 512, 1)):
    inputs, cielo_limpio_predicho = build_unet(input_shape)
    cielo_sucio_predicho = FFTPhysicsLayer(dirty_beam)(cielo_limpio_predicho)
    modelo = models.Model(inputs=inputs, outputs=cielo_sucio_predicho)
```
Esta función conecta las tuberías principales. Crea el punto de entrada, lo pasa por la U-Net, conecta la U-Net a la Capa de Física y declara que la "Salida" oficial de la inteligencia artificial será el *Cielo Sucio Simulado*.

```python
    modelo.compile(optimizer='adam', loss='logcosh')
```
La preparación final del modelo para aprender:
- `optimizer='adam'`: Es el cerebro directivo. Evalúa matemáticamente hacia dónde ajustar los millones de neuronas para que el error sea menor en cada iteración.
- `loss='logcosh'`: La **función de pérdida**. Es la métrica con la que castigamos a la IA por equivocarse. El paper justifica el uso de la métrica Log-Cosh (en lugar de errores cuadráticos comunes) porque es increíblemente estable y robusta ante valores astronómicos atípicamente brillantes, asegurando que la IA no colapse matemáticamente cuando una galaxia super-luminosa aparezca en escena.
