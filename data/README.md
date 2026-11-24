# Datos

| Fuente | Qué medimos | Selección |
| --- | --- | --- |
| [ADE20K SceneParse150](https://ade20k.csail.mit.edu/) | Pared y suelo contra anotación humana | Fotos reales de validación. En la primera muestra se pide al menos un 8 % de área anotada de cada clase. |
| Vídeos propios con ArUco | Estabilidad, detecciones válidas, tiempo y recuperación | Grabar clips separados de calibración, ajuste y evaluación. |
| [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | Error de profundidad en interiores | Usar la descarga etiquetada oficial de unos 2,8 GB, que contiene 1.449 pares RGB y profundidad alineados. Empezar con unas pocas escenas y fijar después un manifiesto por escena. |
| [TUM RGB-D](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/) | Trayectoria si se desarrolla seguimiento sin marcador | Una secuencia con trayectoria de referencia. Su uso depende de que el seguimiento produzca una trayectoria comparable. |

Para NYU, exportar `rawDepths` a metros como matrices 2D alineadas con RGB y conservar los huecos. Registrar nombre de escena, índice original y partición en el manifiesto. Comprobar visualmente la orientación al leer el fichero MATLAB. Mantener habitaciones separadas entre ajuste y prueba. El modelo métrico se evaluará sin ajustarlo con el mapa de referencia de cada foto. Un modelo relativo alineado con ese mapa solo sirve para una comparación de forma y debe informarse por separado.

Antes de congelar el modelo, documentar los datos de entrenamiento de sus pesos y su posible solapamiento con el conjunto de prueba. Revisar también condiciones de uso y atribución. Guardar capturas y datos pesados en almacenamiento local o compartido con el grupo, con los permisos correspondientes.

Primer modelo candidato: [Depth Anything V2 Metric Small para interiores](https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth/README.md). Los autores indican que el ajuste métrico de estos pesos se hace con Hypersim y que la salida se expresa en metros. Medir su velocidad y error antes de decidir si sirve para vídeo.

Para la fase de fotos, guardar el [ZIP de SceneParse150 servido por MIT](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) en `data/raw/ADEChallengeData2016.zip`. Se lee directamente desde el ZIP. El archivo usa 0 como píxel sin etiqueta, 1 para pared y 4 para suelo. La primera referencia es [SegFormer B0 preentrenado en ADE20K](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512). La inferencia registra la revisión del modelo en los resultados.

