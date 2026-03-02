# Analizador de Espectro de Audio 🎵

Este proyecto es un analizador de audio profesional desarrollado en Python, diseñado para extraer y visualizar las componentes de frecuencia de cualquier señal acústica con una estética moderna y precisión matemática.

## 🚀 Características
- **Análisis Selectivo**: Permite cargar fragmentos específicos de archivos de audio (.mp3, .wav, etc.).
- **Procesamiento DSP**: Implementa Transformada Rápida de Fourier (FFT) y ventaneo de Hann.
- **Visualización Dual**: Gráficas sincronizadas del dominio del tiempo y del dominio de la frecuencia.
- **Detección Automática**: Identifica las 3 frecuencias fundamentales más prominentes en el espectro.
- **Estética Premium**: Interfaz visual optimizada con modo oscuro y discretización de datos.

---

## 🧠 Fundamentos Matemáticos

El funcionamiento del código se basa en los pilares del Procesamiento Digital de Señales (DSP):

### 1. Muestreo y Tiempo ($x[n]$)
El audio digital consiste en muestras discretas tomadas a una frecuencia de muestreo ($f_s$). El tiempo asociado a cada muestra $n$ se calcula como:
$$t[n] = \frac{n}{f_s} + t_{inicio}$$
Donde $t_{inicio}$ es el punto de partida seleccionado en el archivo original.

### 2. Ventaneo de Hann
Para evitar el **Spectral Leakage** (fuga espectral) causada por discontinuidades en los bordes del fragmento, aplicamos una ventana de Hann:
$$w[n] = 0.5 \left( 1 - \cos\left( \frac{2\pi n}{N-1} \right) \right)$$
Esto suaviza la señal en los extremos, mejorando la definición de los picos en el espectro.

### 3. Transformada Rápida de Fourier (FFT)
Convertimos la señal del dominio del tiempo al dominio de la frecuencia usando el algoritmo FFT, que resuelve la DFT de manera eficiente:
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} kn}$$
El espectro de magnitud resultante nos permite identificar qué frecuencias (notas o tonos) están presentes.

### 4. Resolución y Nyquist
- **Teorema de Nyquist**: La frecuencia máxima representable es $f_s / 2$.
- **Resolución**: A mayor duración del fragmento, mejor resolución en Hertz ($\Delta f = f_s / N$).

---

## 🛠️ Funcionamiento del Código

El script `audio_analyzer.py` sigue este flujo de trabajo:

1.  **Carga e Indexación**: Usa `librosa` para cargar el audio y extraer el fragmento exacto basado en el tiempo.
2.  **Pre-procesamiento**: Aplica la ventana de Hann si está activada para limpiar la señal.
3.  **Cálculo de FFT**: Obtiene el espectro complejo y calcula su magnitud absoluta.
4.  **Discretización (Binning)**: Agrupa las frecuencias en "bins" (rangos de 2Hz) para suavizar la gráfica y reducir el ruido visual.
5.  **Detección de Picos**: Utiliza `scipy.signal.find_peaks` para localizar los armónicos dominantes.
6.  **Normalización**: Escala la magnitud de 0 a 1 para una visualización comparativa clara.
7.  **Visualización**: Genera una figura de Matplotlib con estética *dark mode*.

---

## 📦 Instalación y Requisitos

Asegúrate de tener Python 3.8+ instalado. Puedes instalar las librerías necesarias con:

```bash
pip install numpy librosa matplotlib scipy
```

## 📖 Uso rápido

Para ejecutar el análisis por defecto:

```python
from audio_analyzer import analyze_audio_fragment

# Analiza el audio "musica.mp3" desde el segundo 190 durante 6 segundos
analyze_audio_fragment("tu_archivo_de_audio.mp3")
```

---
*Desarrollado para el curso de Análisis de Señales.*
