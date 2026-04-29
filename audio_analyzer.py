import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def analyze_audio_fragment(file_path, start_time=190.0, duration=6.0, use_window=True, xlim_hz=5000):
    """
    
    """
    try:
        # Etapa 1: Carga y fragmento
        y, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Nombre del archivo para el título
        song_name = os.path.basename(file_path)
        
        start_idx = int(start_time * sr)
        end_idx = int((start_time + duration) * sr)
        
        if start_idx >= len(y):
            raise ValueError(f"El inicio ({start_time}s) está fuera del rango.")
        
        fragment = y[start_idx:end_idx]
        N = len(fragment)
        t = (np.arange(N) / sr) + start_time
        
        # Ventana de Hann para suavizar
        if use_window:
            fragment_processed = fragment * np.hanning(N)
        else:
            fragment_processed = fragment
            
        # Etapa 2: FFT
        Y = np.fft.fft(fragment_processed)
        freqs = np.fft.fftfreq(N, d=1/sr)
        
        # Máscara más agresiva para eliminar el pico inicial (filtramos hasta 50Hz si es necesario)
        freq_min = 50 
        mask = (freqs >= freq_min) & (freqs <= xlim_hz)
        freqs_raw = freqs[mask]
        mag_raw = np.abs(Y[mask])
        
        # --- DISCRETIZACIÓN (Bins de 5 Hz) ---
        bin_size = 2
        bins = np.arange(freq_min, xlim_hz + bin_size, bin_size)
        bin_indices = np.digitize(freqs_raw, bins) - 1
        
        mag_binned = np.zeros(len(bins) - 1)
        count_binned = np.zeros(len(bins) - 1)
        for i in range(len(mag_raw)):
            idx = bin_indices[i]
            if 0 <= idx < len(mag_binned):
                mag_binned[idx] += mag_raw[i]
                count_binned[idx] += 1
        
        mag_binned = np.divide(mag_binned, count_binned, out=np.zeros_like(mag_binned), where=count_binned > 0)
        freqs_final = (bins[:-1] + bins[1:]) / 2
        
        if mag_binned.max() > 0:
            mag_binned = mag_binned / mag_binned.max()
            
        # --- DETECCIÓN DE 3 FRECUENCIAS FUNDAMENTALES ---
        # Buscamos picos con una distancia mínima para que no sean el mismo armónico
        peaks, properties = find_peaks(mag_binned, height=0.1, distance=20) 
        peak_freqs = freqs_final[peaks]
        peak_mags = mag_binned[peaks]
        
        # Tomar los 3 más altos
        top_indices = np.argsort(peak_mags)[-3:][::-1]
        top_freqs = peak_freqs[top_indices]
        
        # --- ETAPA 3: GRÁFICAS ESTÉTICAS ---
        plt.style.use('dark_background') # Estética moderna
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Análisis: {song_name} | Segmento: {start_time}s - {start_time+duration}s", fontsize=14, color='#00FFCC')
        
        # Panel Izquierdo: Tiempo
        ax1.plot(t, fragment, color='#00CCFF', linewidth=0.3, alpha=0.8)
        ax1.set_title("Onda en el Tiempo [y(u) vs t(s)]", color='white')
        ax1.set_xlabel("Tiempo (s)")
        ax1.set_ylabel("Amplitud")
        ax1.grid(True, color='gray', alpha=0.2)
        
        # Panel Derecho: Fourier
        ax2.fill_between(freqs_final, mag_binned, color='#FF007F', alpha=0.4, label='Espectro')
        ax2.plot(freqs_final, mag_binned, color='#FF007F', linewidth=1)
        
        # Label con las 3 frecuencias fundamentales
        freq_label = "Frecuencias Principales:\n" + "\n".join([f"• {f:.1f} Hz" for f in top_freqs])
        ax2.text(0.95, 0.95, freq_label, transform=ax2.transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, edgecolor='#FF007F'))
        
        ax2.set_title(f"Espectro de Magnitud Discretizado", color='white')
        ax2.set_xlabel("Frecuencia (Hz)")
        ax2.set_ylabel("Magnitud (Normalizada)")
        ax2.set_xlim(freq_min, xlim_hz)
        ax2.grid(True, color='gray', alpha=0.2)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    PATH_AUDIO = "Prince - Purple Rain (Lyrics).mp3"
    analyze_audio_fragment(file_path=PATH_AUDIO)
