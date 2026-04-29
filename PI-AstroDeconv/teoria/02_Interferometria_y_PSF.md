# Interferometría y el "Dirty Beam" (PSF)

## ¿Cómo funciona un radiotelescopio gigante como el SKA?
El telescopio SKA (Square Kilometre Array) no es una sola antena enorme y sólida, sino cientos o miles de antenas repartidas a lo largo de muchos kilómetros de terreno. A esta técnica de combinar múltiples antenas conectadas se le llama **interferometría**.

Al juntar y sincronizar las señales de todas ellas mediante supercomputadoras, actúan de forma equivalente a si tuvieras una sola antena gigante con el mismo diámetro que la distancia máxima entre las antenas más alejadas del arreglo.

## El problema: El "Dirty Beam" (PSF)
Aquí entra la dificultad matemática. Dado que no tenemos miles de kilómetros cuadrados completamente cubiertos de metal, en realidad estamos tomando "muestras" incompletas del cielo. Existen huecos entre cada una de las antenas.

Esto provoca que la imagen que calculamos y obtenemos del cielo no sea perfecta. A este defecto matemático se le llama **Point Spread Function (PSF)** o también **Dirty Beam** (Rayo Sucio). 

**¿Qué significa en la práctica?**
Si apuntamos nuestro radiotelescopio a una sola estrella que debería verse como un punto blanco perfecto, la imagen final no mostrará un solo punto, sino un punto central rodeado de anillos brillantes y borrosos (artefactos). La imagen final (Dirty Image) es matemáticamente una combinación (llamada convolución) de cómo se ve el universo realmente, mezclado con esta "mancha" intrínseca del telescopio (Dirty Beam).

## ¿Por qué usamos WSClean con `niter=0`?
Para arreglar este desenfoque, los astrónomos históricamente han usado un algoritmo tradicional llamado **CLEAN**. Funciona de forma iterativa y lenta: busca puntos muy brillantes y matemáticamente les va restando el patrón del "Dirty Beam" paso a paso.

**WSClean** es una herramienta de software moderna y muy rápida que aplica el algoritmo CLEAN. 

Sin embargo, el objetivo del paper de **PI-AstroDeconv** es justamente NO usar el algoritmo CLEAN tradicional, sino usar Inteligencia Artificial para hacer esta limpieza, haciéndolo más rápido y preciso para detectar la señal del HI (21cm).

Por ello, usamos WSClean configurando un parámetro a cero: **`niter=0`** (cero iteraciones de limpieza). Esto significa que le decimos a WSClean: *"No intentes limpiar la imagen. Sólo genérame la imagen sucia (Dirty Image) y entrégame también el patrón de mancha del telescopio (Dirty Beam)."* Luego, le pasaremos esos datos crudos a nuestra red neuronal para que ella aprenda a hacer el trabajo sucio.
