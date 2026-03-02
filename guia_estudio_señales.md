# Guía de Estudio: Análisis de Señales y la FFT

Esta guía contiene 10 puntos clave basados en la implementación de nuestro analizador de audio en Python para ayudarte a estudiar los conceptos fundamentales del curso.

---

### 1. ¿Qué es la FFT y qué representa físicamente en nuestro código?
**Respuesta:** La FFT (Fast Fourier Transform) es un algoritmo eficiente para calcular la Transformada Discreta de Fourier (DFT). Físicamente, descompone la señal de audio (que está en el dominio del tiempo) en sus componentes de frecuencia. En nuestro código, nos permite pasar de ver "niveles de voltaje/amplitud" a ver "notas musicales o tonos" presentes en el fragmento.

### 2. ¿Para qué sirve la función `np.fft.fftfreq`?
**Respuesta:** Esta función genera el vector de frecuencias correspondiente a los resultados de la FFT. Sin ella, tendríamos los valores de magnitud pero no sabríamos a qué frecuencia (en Hertz) corresponde cada valor. Depende del número de muestras ($N$) y del periodo de muestreo ($d = 1/sr$).

### 3. ¿Por qué aplicamos una ventana de Hann (`np.hanning`) antes de la FFT?
**Respuesta:** Para reducir el fenómeno de **"Spectral Leakage"** (fuga espectral). Una señal recortada abruptamente al inicio y al final introduce frecuencias artificiales por las discontinuidades. La ventana de Hann suaviza los bordes del fragmento a cero, mejorando la resolución y precisión de los picos en el espectro.

### 4. En el código, ¿cómo se escala el eje X de tiempo?
**Respuesta:** Usamos la fórmula `t = np.arange(N) / sr + start_time`. Dividir el índice de la muestra por la frecuencia de muestreo ($sr$) nos da el tiempo relativo, y sumar el `start_time` nos permite posicionar el fragmento en su lugar correcto dentro de la duración total del archivo original.

### 5. ¿Qué significa "normalizar la magnitud" y por qué lo hicimos?
**Respuesta:** Significa dividir todos los valores de magnitud por el valor máximo encontrado (`mag / mag.max()`). Lo hicimos para que el eje Y del espectro siempre esté en un rango de 0 a 1 (o 0% a 100%), facilitando la comparación de la importancia relativa de las frecuencias sin importar qué tan alto está el volumen original del audio.

### 6. ¿Qué es la "frecuencia de Nyquist" y cómo afecta a nuestro `xlim_hz`?
**Respuesta:** La frecuencia de Nyquist es la mitad de la frecuencia de muestreo ($sr/2$). Es la frecuencia máxima que podemos representar digitalmente sin *aliasing*. Si nuestro audio tiene $sr = 44100$ Hz, Nyquist es $22050$ Hz. Por eso, elegir un `xlim_hz` de 5000 o 10000 Hz es seguro, ya que está dentro de los límites físicos de la señal capturada.

### 7. ¿Por qué eliminamos las frecuencias por debajo de 50 Hz en la última versión?
**Respuesta:** Cerca de 0 Hz suelen aparecer picos muy altos llamados "componente DC" (offset) o ruidos de baja frecuencia (hum) que no son parte de la melodía. Al eliminarlos, permitimos que la normalización se enfoque en las frecuencias musicales (armónicos), haciendo que la gráfica de Fourier sea mucho más legible.

### 8. ¿En qué consiste la "discretización en bins" que implementamos?
**Respuesta:** Consiste en agrupar múltiples puntos de la FFT en rangos (ej. cada 2 o 5 Hz) y promediar su valor. Esto reduce la "varianza" o el ruido visual de la gráfica. En señales largas (como nuestros 6 segundos), hay miles de puntos; los bins actúan como un filtro que suaviza la representación para ver mejor la tendencia general.

### 9. ¿Cuál es el papel de la función `find_peaks` de scipy en el proyecto?
**Respuesta:** Sirve para encontrar máximos locales en el espectro de magnitud. Al comparar la altura de un punto con sus vecinos, identifica dónde están las "frecuencias fundamentales" o armónicos más fuertes, los cuales luego etiquetamos en la interfaz gráfica.

### 10. Si quisiéramos analizar un sonido de solo 20ms frente a uno de 6s, ¿qué cambiaría en la resolución?
**Respuesta:** Un fragmento más largo (6s) nos da una **resolución de frecuencia mucho mayor** (puntos más cercanos entre sí en el eje X de Fourier), pero perdemos resolución temporal (no sabemos *cuándo* ocurrió cada nota dentro de esos 6s). Un fragmento corto (20ms) es mejor para ver cambios rápidos en el tiempo, pero sus picos de frecuencia serán más "anchos" y menos definidos.
