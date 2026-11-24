# Plan técnico y experimental

El decorador empieza con fotografías reales. Primero distingue superficies y permite probar un cuadro o una alfombra. Después añade muebles 3D en una foto. La última etapa mantiene el mueble anclado mientras cambia el punto de vista en vídeos reales.

## Desarrollo y evidencias

| Etapa | Resultado verificable |
| --- | --- |
| 1. Fotos y decoración 2D | Máscaras de pared y suelo en ADE20K, IoU por clase, cuadro y alfombra sobre superficies detectadas y ejemplos en que conviene omitir una pieza. |
| 2. Muebles en 3D sobre fotos | Empezar con NYU Depth V2 y una mesa 3D. Evaluar el plano del suelo, la perspectiva y la ocultación detrás de obstáculos. |
| 3. Realidad aumentada en vídeo | Usar grabaciones propias y la base de proyección con ArUco. Comparar pose cruda y suavizada con distancias medidas y puntos de comprobación independientes del marcador. |
| Entrega | Vídeo con aciertos y fallos, memoria PDF, ZIP del código e instrucciones de instalación y reproducción. |

## Experimentos

- **Pared y suelo.** Comparar píxeles de cada clase con la anotación humana mediante IoU y registrar cuántas fotos admiten cada pieza. La muestra doméstica alterna dormitorios, salones y comedores. Dentro de cada categoría toma por nombre las primeras fotos de validación con al menos un 8 % de cada superficie. Sirve para desarrollo. Para una evaluación más representativa, reservar fotos sin seleccionar por resultado e incluir habitaciones con oclusiones y superficies ausentes.
- **Decoración 2D.** La máscara indica qué píxeles pertenecen a una superficie. La primera versión usa un rectángulo para el cuadro y un trapecio aproximado para la alfombra. Evaluar si las esquinas y la orientación siguen las líneas de la habitación antes de añadir inferencia de perspectiva.

- **Calibración.** Guardar cámara, resolución y error de las vistas de ajuste. Usar imágenes reservadas y medidas físicas para comprobar precisión fuera de la calibración.
- **Anclaje.** Medir porcentaje de fotogramas anclados, duración de las pérdidas, tiempo de recuperación y latencia de proceso. Medir temblor con cámara quieta y error contra puntos físicos de comprobación cuando la cámara se mueve. El error de las cuatro esquinas usadas para ajustar la pose mide consistencia, no precisión independiente.
- **Filtrado temporal.** Procesar el mismo vídeo dos veces. Comparar dispersión en tramos quietos y retraso o error en movimiento. Fijar los parámetros en clips de desarrollo y mantenerlos en prueba.
- **Profundidad y ocultación.** Evaluar MAE, RMSE, error relativo y porcentaje de píxeles válidos. Comparar máscaras visibles usando como referencia la profundidad medida y contabilizar huecos. La ocultación exige profundidad real y virtual expresadas en el mismo sistema de cámara.

## Cobertura por demostrar

| Bloque | Evidencia candidata |
| --- | --- |
| BT1 | Reproducción con instrucciones y configuración, figuras y demo dinámica. |
| BT2 | Calibración, pose y proyección 3D. Añadir rectificación con error de medida si contribuye a verificar la colocación. |
| BT3 | Comparar filtrados de profundidad y limpieza de máscara antes de la ocultación, con una alternativa propia si aporta valor. |
| BT4 | Comparar detección directa con seguimiento corto de puntos del entorno, validando pérdidas y deriva. |
| BT5 | Métricas del modelo aprendido. Añadir segmentación CNN solo si mejora selección de suelo u obstáculos. |
| BT6 | Estimación monocular y su aplicación a ocultación. Medir antes de añadir un segundo modelo. |
| BT7 | Ajuste robusto frente a ajuste ordinario sobre correspondencias adicionales o puntos del plano. Cuatro esquinas por sí solas no permiten demostrar un consenso robusto. |

El umbral de evaluación continua es el 100 % de tópicos según el enunciado docente. La tabla identifica trabajo previsto. El cómputo exacto se cerrará con las evidencias y con el criterio del profesor sobre la práctica previa de geometría.

## Estructura del código

- `geometry.py`: cámara, proyección, error y estimación de pose.
- `calibration.py` y `marker.py`: tablero de calibración y detección ArUco.
- `render.py`: mueble paramétrico y proyección de caras.
- `video.py`: lectura, estado del anclaje, filtrado y exportación.
- `demo.py` y `depth.py`: secuencia reproducible y evaluación métrica.

El trabajo previo de [Entrega-Via](https://github.com/PepeGdlT/Entrega-Via) aporta referencias de calibración, flujo óptico y proyección. Si se adapta código, registrar la procedencia y distinguirlo de la aportación nueva.
