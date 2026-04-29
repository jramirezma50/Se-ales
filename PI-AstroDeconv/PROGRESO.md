# Checklist de Progreso: Proyecto PI-AstroDeconv

- [x] Crear estructura de carpetas (`data/raw`, `data/processed`, `models`, `scripts`, `teoria`).
- [x] Crear entorno técnico e instalar dependencias (`requirements.txt`).
- [x] Redactar teoría básica:
  - [x] `01_Sena_21cm.md` (Qué es el hidrógeno neutro).
  - [x] `02_Interferometria_y_PSF.md` (Dirty Beam y WSClean).
  - [x] `03_Arquitectura_IA.md` (U-Net y Convolución FFT).
- [x] Primer script de datos astronómicos (`scripts/01_check_fits.py`).
- [x] Entender y simular la generación de datos (OSKAR / 21cmFAST).
- [x] Preprocesamiento de datos y obtención del Dirty Beam (PSF).
- [ ] Implementar la arquitectura U-Net base.
- [ ] Modificar U-Net para incluir la capa de convolución física (FFT).
- [ ] Entrenamiento del modelo de forma semi-supervisada.
- [ ] Evaluar y extraer métricas (Power Spectrum de 21cm).
