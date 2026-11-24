# Decorador AR para interiores

Miniproyecto de Visión Artificial: colocación y visualización de objetos 3D en interiores mediante profundidad, geometría de cámara y realidad aumentada en vídeo.

El plan técnico y experimental está en [docs/PLAN.md](docs/PLAN.md). La propuesta inicial está disponible en [español](docs/propuesta_inicial.pdf) e [italiano](docs/proposta_iniziale_IT.pdf).

## Empezar con fotos reales

El [notebook de diagnóstico](notebooks/01_diagnostico_colocacion.ipynb) muestra la foto, los errores de la máscara, las referencias de perspectiva, las esquinas editables y la proyección con los módulos del proyecto.

La primera etapa distingue pared y suelo. Para decorar, el modo guiado recibe las cuatro esquinas de cada pieza, marcadas siguiendo las líneas de la habitación. El programa proyecta la textura y descarta las ubicaciones que contradicen la máscara de la superficie. Sin esquinas, exporta las máscaras. El ensayo de colocación automática aproximada está disponible con `--experimental`.

```powershell
python -m pip install -r requirements-photo.txt
$env:PYTHONPATH = 'src'
python -m decorador_ar photo habitacion.jpg results/mi-habitacion --planes ubicaciones.json
python -m decorador_ar photo-study data/raw/ADEChallengeData2016.zip results/ade-estudio --count 8
```

El estudio toma fotos reales de validación del ZIP de ADE20K y guarda un CSV con IoU por superficie, las colocaciones que pudo realizar y la proporción de píxeles colocados sobre la clase correcta. Usa `--offset` para evaluar otro tramo de la muestra. Una foto sola admite `--annotation` para comparar con un PNG indexado ADE y `--art` y `--rug` para suministrar texturas propias.

El JSON admite `wall` y `floor`. Indica los píxeles en orden superior izquierda, superior derecha, inferior derecha e inferior izquierda. Hay dos ejemplos correspondientes a fotos concretas de ADE20K en [assets/planes](assets/planes). La prueba por lotes sigue midiendo el modo experimental.

## Línea de vídeo

El prototipo previo calibra una cámara, estima su posición con ArUco, proyecta una mesa 3D en un vídeo grabado y exporta métricas por fotograma. Queda como base para la última etapa.

## Ejecutar en PowerShell

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
python -m decorador_ar demo results/demo-inicial
```

La carpeta de salida debe estar vacía. Contendrá el vídeo original sintético, la calibración, las dos versiones aumentadas y sendos CSV. La exportación MP4 contiene imagen a frecuencia constante y se escribe sin pista de audio.

## Grabar un primer vídeo real

1. Fotografía un tablero de ajedrez impreso desde al menos diez ángulos y distancias. Usa la misma cámara, modo, resolución y orientación que en el vídeo. Mantén lente y zoom fijos. Mide el lado de una casilla.
2. Genera el marcador. Imprímelo con margen blanco, fíjalo en el suelo y mide el lado del cuadrado negro. Los valores de abajo son ejemplos.
3. Graba un clip corto con una pausa quieta, un movimiento lateral lento, una salida de la referencia y su vuelta.

```powershell
python -m decorador_ar calibrate data/raw/calibracion results/camera.npz --square-m 0.025
python -m decorador_ar marker results/aruco.png
python -m decorador_ar video data/raw/salon.mp4 results/camera.npz results/salon-ar.mp4 --side 0.20 --size 0.65 0.42 0.44 --xy 0.34 0.12 --yaw 10
```

Las dimensiones se dan en metros. `--xy` coloca el centro de la mesa respecto al centro del marcador. `--alpha 0.55` activa el filtro temporal. Si falta el marcador o su ajuste excede el umbral, el objeto desaparece hasta una nueva detección válida.

`depth-score` acepta dos mapas registrados de profundidad en formato `.npy` y en metros. Produce MAE, RMSE, error relativo absoluto y cobertura en JSON. Consulta [el protocolo de datos](data/README.md).

## Estructura

```text
assets/       Recursos visuales y modelos 3D autorizados
data/         Instrucciones para obtener y preparar los datos
docs/         Propuesta, decisiones y documentación
notebooks/    Exploración y experimentos reproducibles
results/      Resultados generados; no datos originales pesados
src/          Código fuente
tests/        Comprobaciones automáticas
```

## Referencias iniciales

- Material oficial de la asignatura Visión Artificial, curso 2026/2027.
- Trabajo previo reutilizable: [`PepeGdlT/Entrega-Via`](https://github.com/PepeGdlT/Entrega-Via).

