# Validación de la decoración en fotos

Esta página documenta el ensayo **experimental**. Sus métricas miden superficie, no calidad visual. En el modo guiado actual se probaron dos fotos adicionales o repetidas de ADE20K con esquinas escogidas visualmente: `ADE_val_00000506` para el cuadro y `ADE_val_00001517` para la alfombra. Ambas piezas quedaron íntegramente en su clase anotada.

Se usaron fotos de validación del ZIP oficial ADE20K SceneParse150. La selección alterna dormitorios, salones y comedores con al menos un 8 % anotado de pared y de suelo. La anotación humana se usa para medir el resultado, después de que el programa haya elegido la ubicación a partir de la foto.

| Muestra | Cuadros colocados | Cuadro en pared | Alfombras colocadas | Alfombra en suelo |
| --- | ---: | ---: | ---: | ---: |
| Versión inicial, 9 fotos | 9/9 | 74,4 % | 9/9 | 98,4 % |
| Versión revisada, mismas fotos | 6/9 | 100 % | 7/9 | 100,0 % |
| Versión revisada, 18 fotos reservadas | 12/18 | 99,6 % | 11/18 | 99,9 % |

Los porcentajes son el promedio por pieza medible, sobre sus píxeles con anotación válida. La versión inicial midió el polígono completo. La revisada mide los píxeles donde la pieza tiene al menos un 50 % de opacidad. En la reserva, el peor cuadro obtuvo 96,3 % y la peor alfombra 98,5 %. Contar las omisiones evita confundir precisión con capacidad de decorar todas las habitaciones.

La revisión visual incluye un cuadro que antes invadía una ventana (ADE_val_00000117), un salón con ambas piezas (ADE_val_00000506) y salones y comedores de reserva. La alfombra sigue un plano aproximado y el cuadro se trata como frontal. La prueba de solapamiento comprueba la superficie visible, pero no certifica escala física, estilo o alineación cuando la cámara mira la pared de lado.

Para repetir con el ZIP oficial en el directorio mostrado:

```powershell
$env:PYTHONPATH = 'src'
python -m decorador_ar photo-study data/raw/ADEChallengeData2016.zip results/repeticion-ejemplos --count 9
python -m decorador_ar photo-study data/raw/ADEChallengeData2016.zip results/repeticion-reserva --count 18 --offset 27
```
