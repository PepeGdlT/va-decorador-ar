# Decorador AR para interiores

[Documentazione in italiano](docs/it/README.md)

Miniproyecto de Visión Artificial para decorar fotos reales de interiores con cuadros, alfombras y muebles 3D.

El plan técnico y experimental está en [docs/PLAN.md](docs/PLAN.md). La propuesta inicial está disponible en [español](docs/propuesta_inicial.pdf) e [italiano](docs/proposta_iniziale_IT.pdf).

## Abrir en PyCharm

Abre la carpeta raíz `va-decorador-ar`. En **Settings → Project → Python Interpreter**, selecciona el intérprete de `.venv\Scripts\python.exe` si PyCharm no lo detecta. En una instalación nueva puedes preparar el entorno desde la terminal del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pycharm.txt
```

En el desplegable junto a ▶ están las configuraciones numeradas. Cada ejecución crea su propia carpeta dentro de `results/` e imprime su ruta en la consola.

| Configuración | Qué ejecuta |
| --- | --- |
| 01 - Ejecutar notebook principal | Ejecuta diseños 2D, pruebas de muebles 3D y evaluación ADE20K si está disponible. Guarda una copia con salidas en `decorador-ejecutado.ipynb`. |
| 02 - Probar foto guiada de ADE20K | Extrae una foto del ZIP, segmenta las superficies y aplica la colocación guiada del cuadro. |
| 03 - Estudiar 9 fotos de ADE20K | Ejecuta el ensayo automático experimental y escribe un CSV de resultados. |
| 04 - Demo AR sintética | Genera la secuencia, los vídeos aumentados y los CSV de vídeo. |
| 05 - Generar marcador ArUco | Guarda un marcador PNG imprimible. |

Las configuraciones 02 y 03 requieren `data/raw/ADEChallengeData2016.zip` ([descarga y notas](data/README.md)). Para usar el editor interactivo, ejecuta el notebook principal abierto en PyCharm.

## Empezar con fotos reales

Abre [Decorador_AR.ipynb](notebooks/Decorador_AR.ipynb) con el kernel de `.venv`. El editor ofrece veinte vistas reales de ZInD. El cuaderno enseña seis diseños iniciales, los apoyos de cocina y una galería de las vistas nuevas. La primera ejecución prepara las vistas. El notebook de diagnóstico conserva el ensayo de ADE20K.

En **Componer una habitación**, elige habitación, superficie y pieza. Mueve el ratón para ver la pieza translúcida y haz clic para fijarla. **Ajustes** contiene la escala, el giro y la luz. Si PyCharm oculta la vista integrada, abre el enlace del navegador o copia la dirección local que imprime la celda. El editor funciona mientras el kernel esté activo. **Guardar PNG** guarda las piezas confirmadas de la vista actual.

Las piezas pueden continuar fuera de la foto. La parte que queda dentro debe encajar en la pared, el suelo o el apoyo elegido.

Para probar los apoyos, elige **cocina_01 → En altura → Detectar apoyos**. El contorno cian señala una zona de la encimera y **Apoyo** rellena el centro. También puedes elegir los tableros de los muebles virtuales o marcar cuatro esquinas y una altura relativa a la cámara. La detección combina segmentación y profundidad estimada. Necesita una referencia visible del suelo y puede omitir apoyos estrechos, tapados o mal segmentados. Con **Luz y sombra** puedes comparar el ajuste de brillo y color local y las sombras de contacto de las piezas 3D.

El apartado **Cobertura del enunciado** relaciona los experimentos con los BT. Incluye la comparación de máscaras con ADE20K y el registro de un cuadro entre dos vistas con ORB y RANSAC. Para omitir la descarga de la CNN de comparación, pon `COMPARE_CNN = False` en la sección de segmentación.

## Procesar fotos por terminal

La primera etapa distingue pared y suelo. Para decorar, el modo guiado recibe las cuatro esquinas de cada pieza, marcadas siguiendo las líneas de la habitación. El programa proyecta la textura y descarta las ubicaciones que contradicen la máscara de la superficie. Sin esquinas, exporta las máscaras. El ensayo de colocación automática aproximada está disponible con `--experimental`.

```powershell
python -m pip install -r requirements-photo.txt
$env:PYTHONPATH = 'src'
python -m decorador_ar photo habitacion.jpg results/mi-habitacion --planes ubicaciones.json
python -m decorador_ar photo-study data/raw/ADEChallengeData2016.zip results/ade-estudio --count 8
```

El estudio toma fotos reales de validación del ZIP de ADE20K y guarda un CSV con IoU por superficie, las colocaciones que pudo realizar y la proporción de píxeles colocados sobre la clase correcta. Usa `--offset` para evaluar otro tramo de la muestra. Una foto sola admite `--annotation` para comparar con un PNG indexado ADE y `--art` y `--rug` para suministrar texturas propias.

Para estudios grandes, añade `--metrics-only`. Conserva el mismo CSV y evita generar copias de las imágenes y montajes. El CSV guarda cada fila conforme termina la foto. `estudio.json` registra el tramo de validación y la revisión del modelo.

El JSON admite `wall` y `floor`. Indica los píxeles en orden superior izquierda, superior derecha, inferior derecha e inferior izquierda. Hay dos ejemplos correspondientes a fotos concretas de ADE20K en [assets/planes](assets/planes). La prueba por lotes sigue midiendo el modo experimental.

## Experimentos de vídeo aparte

El prototipo previo de ArUco y la reproyección con las poses de TUM se conservan como experimentos independientes. El notebook principal se centra en fotografías.

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
python -m decorador_ar demo results/demo-inicial
```

La carpeta de salida debe estar vacía. Contendrá el vídeo original sintético, la calibración, las dos versiones aumentadas y sendos CSV. La exportación MP4 contiene imagen a frecuencia constante y se escribe sin pista de audio.

### Grabar un primer vídeo real

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

## Ejecutar las pruebas

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

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

