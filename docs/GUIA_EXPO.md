# Guion breve para la exposición

## Alcance que presentamos

El MVP es un **editor fotográfico de interiores**, no una aplicación de AR de vídeo. SegFormer propone las máscaras de pared y suelo; las referencias geométricas de ZInD permiten rectificar esos planos; el editor coloca decoraciones 2D y muebles procedurales 3D sobre la fotografía.

La propuesta inicial incluía vídeo y cámara en movimiento. El prototipo que podemos enseñar de forma reproducible se ha centrado en fotografías. Conviene decirlo al principio y presentar vídeo, calibración propia y oclusión con profundidad real como trabajo pendiente.

## Demostración en directo (3–4 minutos)

1. **Foto y superficies.** Abre una de las vistas `bonus` y muestra brevemente las máscaras de pared/suelo. Explica que la segmentación ayuda a aceptar la superficie, mientras que la geometría del plano de esta demo procede de anotaciones conocidas.
2. **Composición 2D.** Coloca un cuadro y una alfombra, ajusta posición y escala y muestra la perspectiva. Aclara que no es una colocación geométrica automática para cualquier foto.
3. **Mueble 3D.** Coloca el sofá o la mesa baja. Describe los modelos como muebles procedurales low-poly. Gira o escala la pieza y compara luz/sombra.
4. **Apoyo real.** En `cocina_01`, muestra la propuesta de zona de apoyo y un objeto elevado. Depth Anything aporta una estimación monocular para proponer el apoyo; no se ha medido su error métrico contra RGB-D.
5. **Editor y exportación.** Añade y elimina una pieza, y guarda un PNG. Abre el PNG exportado para demostrar que el resultado no depende solo de la vista del editor.
6. **Evidencia.** Enseña IoU ADE20K y el experimento ORB/RANSAC con giro conocido; usa la tabla de sensibilidad RANSAC recién generada. Distingue siempre los resultados medidos de las limitaciones.

## Evidencia y límites

- ADE20K: IoU de pared y suelo frente a anotaciones; la muestra es pequeña y no equivale a evaluación general.
- Registro: ORB y homografía RANSAC entre dos vistas generadas con giro conocido. Los puntos de control provienen de la geometría conocida; no son una prueba de traslación.
- Muebles: render procedural y profundidad entre caras virtuales. No afirmar oclusión de muebles detrás de objetos reales: el editor no pasa un mapa de profundidad medido al render.
- Traslación: las dos fotos propias no proporcionan puntos de referencia independientes ni una estimación validada. No usar la figura actual como resultado de precisión.
- Cobertura BT: el porcentaje es una propuesta de tópicos, no una calificación certificada. Confirmar con el enunciado oficial y el profesor.

## Respaldo si falla la demo en directo

Ejecuta el notebook principal antes de exponer y conserva la copia ejecutada y sus PNG/CSV de la carpeta de resultados. La salida de `results/` se ignora en Git; cópiala al espacio compartido del equipo o a la presentación. No dependas de una descarga de modelos durante la exposición.
