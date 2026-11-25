# Plan técnico y experimental

El decorador se centra en fotografías reales. Primero distingue superficies y permite probar cuadros y alfombras. Después añade muebles 3D en la misma foto. Los ensayos de vídeo quedan como ampliación aparte.

## Desarrollo y evidencias

| Etapa | Resultado verificable |
| --- | --- |
| 1. Fotos y decoración 2D | El [notebook principal](../notebooks/Decorador_AR.ipynb) muestra máscaras, rectificación de pared y suelo y colocación en seis vistas de ZInD. ADE20K aporta el estudio de IoU por clase. Para cerrar la etapa, medir geometría con referencias independientes y registrar fallos de colocación. |
| 2. Muebles en 3D sobre fotos | El editor permite combinar y previsualizar piezas. Proyecta muebles sobre el suelo y detecta zonas para colocar objetos sobre una encimera real. Las piezas 3D reciben ajustes de brillo local y sombras de contacto. |
| Ampliación de vídeo | Se conservan las pruebas de TUM RGB-D y ArUco como experimentos independientes. |
| Entrega | Memoria PDF, código y guía de reproducción. En continua, el vídeo breve de demostración es opcional. En global se exige un vídeo de presentación. |

## Experimentos

- **Pared y suelo.** Comparar píxeles de cada clase con la anotación humana mediante IoU y registrar cuántas fotos admiten cada pieza. La muestra doméstica alterna dormitorios, salones y comedores. Dentro de cada categoría toma por nombre las primeras fotos de validación con al menos un 8 % de cada superficie. Sirve para desarrollo. Para una evaluación más representativa, reservar fotos sin seleccionar por resultado e incluir habitaciones con oclusiones y superficies ausentes.
- **Decoración 2D.** Con las aristas anotadas de ZInD se construyen dos homografías. La forma y el giro de cada pieza se definen en su plano. SegFormer comprueba cuánto del objeto queda sobre la clase esperada. Las seis vistas vienen de una vivienda y sirven para desarrollo. Para medir precisión geométrica, marcar líneas o esquinas independientes en otras fotos y comparar desviación angular y error en píxeles. La ida y vuelta de la misma homografía comprueba consistencia numérica.

- **Calibración.** Guardar cámara, resolución y error de las vistas de ajuste. Usar imágenes reservadas y medidas físicas para comprobar precisión fuera de la calibración.
- **Anclaje.** Medir porcentaje de fotogramas anclados, duración de las pérdidas, tiempo de recuperación y latencia de proceso. Medir temblor con cámara quieta y error contra puntos físicos de comprobación cuando la cámara se mueve. El error de las cuatro esquinas usadas para ajustar la pose mide consistencia, no precisión independiente.
- **Filtrado temporal.** Procesar el mismo vídeo dos veces. Comparar dispersión en tramos quietos y retraso o error en movimiento. Fijar los parámetros en clips de desarrollo y mantenerlos en prueba.
- **Profundidad y ocultación.** Evaluar MAE, RMSE, error relativo y porcentaje de píxeles válidos. Comparar máscaras visibles usando como referencia la profundidad medida y contabilizar huecos. La ocultación exige profundidad real y virtual expresadas en el mismo sistema de cámara.

## Cobertura y evidencias

Tópicos candidatos según la sección 6 de *Enunciados de Prácticas* 2026/2027. Los pesos representan el objetivo de trabajo. El estado describe la evidencia ejecutable y lo que falta para defender cada tópico.

| Bloque | Candidatos | Evidencia disponible | Para cerrar |
| --- | --- | --- | --- |
| BT1 | BT1a, BT1b, BT1c · 15 % | Versiones, visualizaciones y editor por cursor. | Reservar un manifiesto fuera de la vivienda de desarrollo. |
| BT2 | BT2a, BT2b, BT2e, BT2f · 20 % | Homografías, rectificación, transferencia entre dos vistas con giro conocido y proyección 3D. | Validar calibración y pose con fotos propias y repetir el registro con traslación. |
| BT3 | BT3a, BT3d, BT3g · 20 % | Sombra suavizada, regiones de apoyo y comparación de morfología NumPy con OpenCV. | Medir cómo cambia la detección de apoyos sin morfología. |
| BT4 | BT4a, BT4b, BT4d · 15 % | ORB, correspondencias y traslado de la huella del cuadro. | Ampliar la comparación a otras habitaciones. |
| BT5 | BT5d, BT5f, BT5h · 15 % | IoU, comparación con una CNN, mapas de probabilidad y umbralización sobre ADE20K. | Ampliar la muestra y medir el tiempo por modelo. |
| BT6 | BT6a, BT6f · 10 % | SegFormer y profundidad monocular aplicada a apoyos reales. | Medir profundidad y error de altura con RGB-D. |
| BT7 | BT7a, BT7c · 10 % | RANSAC frente al ajuste sin rechazo de correspondencias. | Analizar sensibilidad del umbral e incluir más pares de vistas. |

La selección con evidencia en el cuaderno suma **105 %**. El enunciado exige **100 % en evaluación continua** y **120 % en evaluación global**. La cobertura está sujeta al criterio del profesor.

**Orden de integración.** Validar el apoyo y la perspectiva en más fotos. Después, probar estabilidad con dos puntos de vista de la misma estancia e integrar ocultación con profundidad RGB-D.

**Cierre con alcance fotográfico**

1. Reservar fotos de otras viviendas y medir apoyo y perspectiva con referencias independientes. Comparar la máscara original con una limpieza morfológica y registrar IoU, falsos rechazos y ejemplos difíciles.
2. Registrar un plano entre dos fotos tomadas desde posiciones distintas. Comparar ORB con ajuste robusto y ordinario usando error en puntos de comprobación, además de los inliers.
3. Medir el modelo de profundidad sobre fotos RGB-D y comprobar si mejora la ocultación del mueble detrás de objetos reales. Para los apoyos automáticos, anotar zonas utilizables y alturas en viviendas reservadas y medir aciertos, omisiones y error de altura. Para la luz, comparar las sombras de contacto con una referencia fotografiada antes de añadir sombras direccionales.

## Estructura del código

- `photo.py` y `photo_study.py` gestionan la segmentación y su evaluación con ADE20K.
- `panorama.py` prepara las vistas de ZInD y traduce sus anotaciones al sistema de cámara.
- `rectification.py` define la geometría de los planos y la colocación con perspectiva.
- `placement.py` comparte composición y medidas de encaje. `composition.py` conserva las piezas y `editor_web.py` conecta el editor del notebook con la colocación.
- `geometry.py`: cámara, proyección, error y estimación de pose.
- `calibration.py` y `marker.py`: tablero de calibración y detección ArUco.
- `render.py`: mueble paramétrico y proyección de caras.
- `video.py`: lectura, estado del anclaje, filtrado y exportación.
- `demo.py` y `depth.py`: secuencia reproducible y evaluación métrica.

El trabajo previo de [Entrega-Via](https://github.com/PepeGdlT/Entrega-Via) aporta referencias de calibración, flujo óptico y proyección. Si se adapta código, registrar la procedencia y distinguirlo de la aportación nueva.
