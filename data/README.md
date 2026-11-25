# Datos

| Fuente | Qué medimos | Selección |
| --- | --- | --- |
| [ADE20K SceneParse150](https://ade20k.csail.mit.edu/) | Pared y suelo contra anotación humana | Fotos reales de validación. En la primera muestra se pide al menos un 8 % de área anotada de cada clase. |
| Vídeos propios con ArUco | Estabilidad, detecciones válidas, tiempo y recuperación | Grabar clips separados de calibración, ajuste y evaluación. |
| [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | Error de profundidad en interiores | Usar la descarga etiquetada oficial de unos 2,8 GB, que contiene 1.449 pares RGB y profundidad alineados. Empezar con unas pocas escenas y fijar después un manifiesto por escena. |
| [TUM RGB-D](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/) | Reproyección en vídeo real con poses de referencia y profundidad registrada | Dos tramos de `fr1/room`, grabada en una oficina. |
| [Zillow Indoor Dataset (ZInD)](https://github.com/zillow/zind) | Primeras pruebas sencillas de colocación | Se usan seis vistas rectilíneas derivadas del recorrido público de muestra: viviendas reales desocupadas, no el conjunto completo. ZInD es para uso académico no comercial y debe citarse. |

Para NYU, exportar `rawDepths` a metros como matrices 2D alineadas con RGB y conservar los huecos. Registrar nombre de escena, índice original y partición en el manifiesto. Comprobar visualmente la orientación al leer el fichero MATLAB. Mantener habitaciones separadas entre ajuste y prueba. El modelo métrico se evaluará sin ajustarlo con el mapa de referencia de cada foto. Un modelo relativo alineado con ese mapa solo sirve para una comparación de forma y debe informarse por separado.

Antes de congelar el modelo, documentar los datos de entrenamiento de sus pesos y su posible solapamiento con el conjunto de prueba. Revisar también condiciones de uso y atribución. Guardar capturas y datos pesados en almacenamiento local o compartido con el grupo, con los permisos correspondientes.

Primer modelo candidato: [Depth Anything V2 Metric Small para interiores](https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth/README.md). Los autores indican que el ajuste métrico de estos pesos se hace con Hypersim y que la salida se expresa en metros. Medir su velocidad y error antes de decidir si sirve para vídeo.

Para la fase de fotos, guardar el [ZIP de SceneParse150 servido por MIT](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) en `data/raw/ADEChallengeData2016.zip`. Se lee directamente desde el ZIP. El archivo usa 0 como píxel sin etiqueta, 1 para pared y 4 para suelo. La primera referencia es [SegFormer B0 preentrenado en ADE20K](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512). La inferencia registra la revisión del modelo en los resultados.

El [notebook principal](../notebooks/Decorador_AR.ipynb) prepara catorce vistas de una vivienda, seis para la galería 2D, seis para probar muebles y dos de la cocina para probar apoyos elevados. Descarga los panoramas y sus anotaciones del [recorrido de muestra oficial](https://github.com/zillow/zind/tree/main/sample_tour/000) y genera vistas de 960 × 720 en `data/processed/zind-rectified/`. Conserva los originales en `data/raw/zind-sample/`.

**Detectar apoyos** descarga bajo demanda [Depth Anything V2 Metric Indoor Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) en la caché local de Hugging Face. La escala de la predicción se ajusta con el suelo anotado y las alturas se expresan en relación con la altura de la cámara. Ambos ejemplos de cocina pertenecen a la misma vivienda y sirven para probar la interacción.

Para ejecutar el vídeo real, guarda el [archivo oficial fr1/room](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz) como `data/raw/tum-fr1-room.tgz` (782 MB). El código lee los tramos seleccionados desde el archivo comprimido. Los MP4 se escriben a frecuencia constante y el CSV conserva los tiempos originales. El benchmark de Sturm et al. se publica con [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Consulta [el artículo y sus especificaciones](https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf).

