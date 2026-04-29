# PI-AstroDeconv: Arquitectura de la Red Neuronal (U-Net y Convolución FFT)

## ¿Qué es la Red Neuronal U-Net?
La U-Net es una arquitectura de red neuronal muy famosa inventada originalmente para procesar imágenes médicas. Su diseño, como indica su nombre, tiene una forma de "U":

1. **El Encoder (La bajada):** Toma la imagen grande original y mediante múltiples capas ("convoluciones" y agrupaciones) la va comprimiendo. A medida que la imagen se hace más pequeña en resolución, la red aumenta la cantidad de "filtros", extrayendo las características, patrones y contexto más importantes de la imagen.
2. **El Decoder (La subida):** Toma esa información muy comprimida y comienza a agrandarla (upsampling) hasta devolverle el tamaño original de la imagen. La genialidad de U-Net es que, durante esta subida, también recibe "atajos" de información detallada que guardó durante la bajada (llamadas conexiones residuales), permitiendo que reconstruya una imagen limpia con detalles perfectos.

## La magia del Paper: Convolución Informada por Física
Normalmente, a una U-Net le entregaríamos una imagen "sucia" y esperaríamos que nos devuelva una imagen "limpia", comparando su resultado con imágenes perfectas (Ground Truth). 

Sin embargo, en astronomía real no tenemos "imágenes limpias y perfectas" de etiqueta (labels) de antemano. El acercamiento de **PI-AstroDeconv** es brillante porque inserta nuestro conocimiento previo de física dentro de la misma IA:

**La IA no es evaluada por qué tan limpia da la imagen, sino por cómo se ensucia.**
1. La U-Net predice cómo debería ser el "cielo limpio".
2. **Capa Física:** A ese cielo limpio que generó la IA, la red misma le aplica matemáticamente la mancha específica del telescopio (el Dirty Beam que obtenemos de WSClean).
3. Se compara el resultado *ensuciado artificialmente* con la observación cruda que captó el telescopio en la realidad (que también está sucia).

Si el resultado de la U-Net, una vez ensuciado, cuadra perfectamente con lo que observó el telescopio, ¡significa que la U-Net logró adivinar la versión limpia del cielo real! Esto se llama **aprendizaje no supervisado con física (Physics-Informed Semi-Supervised Learning)**.

## ¿Qué es la Convolución FFT (Fast Fourier Transform)?
Ensamblar el "cielo limpio" con el "Dirty Beam" requiere una operación matemática llamada *Convolución*. 

Dado que el Dirty Beam puede tener matrices gigantescas de píxeles (por ejemplo, imágenes de 2048 x 2048 píxeles), hacer una convolución tradicional requeriría multiplicar la mancha, píxel por píxel, deslizando sobre toda la imagen. Esta operación (complejidad $O(n^4)$) paralizaría hasta a las mejores tarjetas gráficas de la computadora durante el entrenamiento.

La solución del paper es usar la **Transformada Rápida de Fourier (FFT)** en la última capa de la red neuronal. 
Un viejo truco matemático dicta que: **La convolución de dos imágenes en el mundo espacial equivale a una simple multiplicación en el "mundo de las frecuencias" (Fourier).**

Así que la red:
1. Pasa el cielo limpio y el Dirty beam al mundo de Fourier usando FFT.
2. Multiplica ambos.
3. Usa la transformada inversa (IFFT) para devolver la imagen al mundo espacial normal.

Este pequeño cambio arquitectónico reduce la complejidad matemática a $O(n^2 \log n)$, haciendo que la red pueda entrenarse miles de veces más rápido.
