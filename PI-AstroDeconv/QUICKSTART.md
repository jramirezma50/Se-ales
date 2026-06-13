# PI-AstroDeconv - Guía de Inicio (Linux)

## 🚀 Inicio Rápido

### 1. Instalar dependencias del sistema (una sola vez)

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev build-essential
```

### 2. Configurar entorno Python

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Verificar instalación

```bash
python3 scripts/01_check_fits.py
```

## 🔧 Configuración en VS Code

VS Code está **listo para usar** en Linux:

✅ Intérprete configurado: `.venv/bin/python`  
✅ Análisis de código habilitado  
✅ Extensiones recomendadas incluidas  

### Extensiones necesarias

VS Code te pedirá instalar extensiones recomendadas cuando abras el proyecto. Acepta la recomendación o instálalas manualmente:

1. `ms-python.python` - Soporte Python oficial
2. `ms-python.vscode-pylance` - Análisis inteligente
3. `ms-toolsai.jupyter` - Para notebooks (opcional)

## 📁 Estructura del Proyecto

```
PI-AstroDeconv/
├── scripts/
│   ├── 01_check_fits.py      # Lee archivos FITS astronómicos
│   ├── 02_simulate_data.py   # Genera datos de entrenamiento
│   └── 03_model_unet.py      # Construye la red neuronal
├── data/
│   └── raw/                  # Archivos .fits y PNG
├── teoria/
│   ├── 01_Sena_21cm.md      # Física del hidrógeno neutro
│   ├── 02_Interferometria_y_PSF.md
│   └── 03_Arquitectura_IA.md
├── requirements.txt          # Dependencias Python
├── PROGRESO.md              # Checklist del proyecto
├── RESUMEN_EXPOSICION.md    # Documentación técnica
└── SETUP_LINUX.md           # Este archivo (instrucciones detalladas)
```

## 🔄 Flujo de Trabajo

Cada script construye sobre el anterior:

```
01_check_fits.py
    ↓ aprende a abrir archivos astronómicos
02_simulate_data.py
    ↓ genera datos realistas para entrenar
03_model_unet.py
    ↓ construye la IA que desconvoluciona imágenes
```

## 🎯 Próximos Pasos

1. ✅ Configuración completada
2. ➡️ Ejecuta `02_simulate_data.py` para generar datos de prueba
3. ➡️ Explora la arquitectura con `03_model_unet.py`
4. ➡️ Implementa el entrenamiento (próxima fase)

## ❓ Solución de Problemas

**P: "ModuleNotFoundError: No module named 'tensorflow'"**  
R: Asegúrate de que el entorno está activado: `source .venv/bin/activate`

**P: "Python interpreter not found"**  
R: Presiona `Ctrl+Shift+P` → "Python: Select Interpreter" → elige `.venv/bin/python`

**P: "Permission denied" con sudo**  
R: Necesitas la contraseña de administrador. Pídela si no la sabes.

## 📚 Documentación Completa

Para detalles técnicos: ver [RESUMEN_EXPOSICION.md](RESUMEN_EXPOSICION.md)  
Para configuración detallada: ver [SETUP_LINUX.md](SETUP_LINUX.md)  
Para progreso del proyecto: ver [PROGRESO.md](PROGRESO.md)
