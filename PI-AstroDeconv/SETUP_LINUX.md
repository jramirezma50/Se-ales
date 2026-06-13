# Configuración del Proyecto en Linux

## Estado actual
- ✅ Python 3.10.12 instalado
- ❌ pip no instalado
- ❌ python3-venv no instalado
- ✅ VS Code configurado para Linux

## Pasos para completar la configuración

### Paso 1: Instalar herramientas del sistema (requiere sudo)

Ejecuta en una terminal:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev build-essential
```

Cuando pida contraseña, escríbela (no verás los caracteres - es normal).

### Paso 2: Crear entorno virtual

Desde la carpeta del proyecto:

```bash
cd ~/Se-ales/PI-AstroDeconv
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar dependencias

Con el entorno activado:

```bash
pip install -r requirements.txt
```

Esto instalará:
- tensorflow (IA y procesamiento de tensores)
- astropy (formato FITS astronómico)
- numpy (matemáticas)
- matplotlib (visualización)
- scipy (procesamiento de señal)

### Paso 4: Verificar instalación

```bash
python3 scripts/01_check_fits.py
```

Debería mostrar un mensaje de prueba.

## Configuración en VS Code

La configuración en `.vscode/settings.json` ya está lista para Linux:
- Intérprete apunta a `.venv/bin/python`
- Linting y formato automático activados
- Black como formateador

Cuando hayas completado los pasos 1-3, VS Code debería detectar automáticamente el entorno.

Si no lo detecta, presiona `Ctrl+Shift+P` y busca "Python: Select Interpreter", luego elige `.venv/bin/python`.

## Extensiones recomendadas en VS Code

Instala estas desde la tienda de extensiones (Ctrl+Shift+X):

1. **Python** (Microsoft)
2. **Pylance** (Microsoft) 
3. **Jupyter** (Microsoft)
4. **GitLens** (Eric Amodio) - opcional pero recomendado

## Alternativa si no puedes ejecutar sudo

Si no puedes ejecutar `sudo` (sin contraseña), contacta al administrador del sistema para que instale:
- `python3-venv`
- `python3-pip`  
- `python3-dev`
- `build-essential`

Una vez estén instalados, sigue los Pasos 2-4 arriba.
