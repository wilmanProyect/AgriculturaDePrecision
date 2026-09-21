# Líneas de siembra y posibles fallas

En **Analizador Agrícola**, selecciona el ortomosaico RGB y las parcelas, y pulsa
**Detectar líneas de siembra y posibles fallas**. Hay dos métodos, elegidos según
si indicas un modelo YOLO en el sub-diálogo de parámetros:

- **Sin modelo (por defecto)**: heurística RGB descrita en "Método heurístico RGB"
  más abajo. No requiere pesos, funciona con cualquier ortomosaico RGB.
- **Con modelo YOLO (.pt)**: cada línea/falla es una instancia que el modelo
  localiza directamente (ver "Método YOLO11 con modelo entrenado" más abajo).
  Soporta tanto segmentación de instancias (máscaras/polígonos) como pose
  (caja + 2 keypoints) — el tipo se detecta solo, según cómo se entrenó el
  modelo cargado. Más robusto ante malezas, sombras y curvas de cabecera una vez
  entrenado con imágenes reales del cultivo, pero depende de la calidad y
  cobertura del dataset de entrenamiento.

El campo "Modelo YOLO (.pt)" trae precargado por defecto el modelo ya entrenado
(`row_lines_pose-2`, ver más abajo) si existe en el equipo — se puede limpiar el
campo para usar la heurística RGB en su lugar, o reemplazarlo por otro `.pt`.

Parámetros compartidos por ambos métodos:

- Separación aproximada entre surcos (0,45 m inicial): la heurística la usa para
  estimar orientaciones; con YOLO se usa para agrupar detecciones que pertenecen
  a la misma fila física al inferir huecos entre ellas (ver más abajo).
- Longitud mínima de falla/línea: 1 m inicialmente.
- **Resolución de análisis**: "Detectar automáticamente" (recomendado, activado por
  defecto) usa el detalle real del ortomosaico — no un valor fijo adivinado. Con
  YOLO, además define el tamaño del recorte que se le pasa al modelo por pieza.
  Desmarcá la casilla para fijar un valor manual en m/píxel. El valor
  efectivamente usado en cada corrida queda en el log del plugin y en
  `resumen.json` (`options.resolution_m`).
- **Lotes grandes** (parcelas de decenas o cientos de hectáreas, ej. 150 ha): se
  dividen automáticamente en piezas más chicas que respeten `max_pixels` (12
  millones de píxeles por pieza por defecto) a la resolución elegida, en vez de
  fallar o perder detalle bajando la resolución. Una hilera que cruce el límite
  entre dos piezas vecinas aparece partida en el resultado (más líneas, más
  cortas) y un hueco justo en ese límite puede no detectarse — no afecta la suma
  total de metros sembrados/no sembrados, que es lo que reporta el resumen.
  Lotes más grandes generan más piezas y tardan más (la barra de progreso y el
  botón Cancelar siguen funcionando igual). Para menos fragmentación a costa de
  más memoria por pieza, subí `RowOptions.max_pixels` (no expuesto en la UI
  todavía).
- Cada ejecución crea una carpeta propia con GeoPackages, estilos QML y resumen
  JSON, dentro de la carpeta elegida. Si una capa no tiene resultados, no se crea.

Solo para la heurística RGB: umbral de vegetación (0,065 inicial, mayor = más
tramos marcados como falla) — con YOLO se ignora, el modelo ya localiza cada
línea directamente.

Las líneas sembradas se muestran en celeste y las posibles fallas en rojo. Las
longitudes están en metros, en un CRS UTM estimado para las parcelas. `parcela_id`
es el orden (desde 1) de las parcelas procesadas — o "N.K" (pieza K de la
parcela N) si esa parcela se dividió por ser grande; `surco_id` enlaza cada
falla con su línea. El resumen JSON incluye `row_length_m` (metros de línea
sembrada) y `gap_length_m` (metros no sembrados), y en el diálogo del plugin se
muestran ambos totales al terminar. Los resultados se cargan al finalizar, desde
el hilo principal de QGIS. La cancelación no publica capas parciales como
resultados completos.

## Método heurístico RGB (sin modelo)

`core/analysis/row_detection.py` (`RowDetector`). Lee ventanas RGB con máscara de
datos válidos, calcula ExG, estima hasta tres orientaciones mediante estructura
local, sigue crestas de vegetación y mide interrupciones internas.

Es un método RGB preliminar, no entrenado. Las malezas, sombras, curvas fuertes y
cabeceras pueden provocar errores. Un espacio sin datos nunca se interpreta como
suelo desnudo. Los vacíos en extremos y los surcos completamente ausentes no se
infieren: falta evidencia para afirmar que debían estar sembrados. Revisa el
resultado en campo o contra anotaciones manuales antes de cuantificar fallas.
Hay un límite de 12 millones de píxeles por parcela para acotar la memoria; divide
parcelas mayores o cambia la resolución respetando la separación entre surcos.

## Método YOLO11 con modelo entrenado

`ai/inference/row_segmenter.py` (`RowSegmenter`). Recorta cada parcela a la
resolución indicada (mismo recorte georreferenciado que usa la heurística, sin el
cálculo de ExG) y corre el modelo cargado. Según su tarea:

- **Segmentación** (`masks`): reduce cada máscara a su línea central
  (`skimage.morphology.skeletonize` + camino más largo del esqueleto vía
  `scipy.sparse.csgraph.dijkstra`, "barrido doble": funciona bien para una única
  curva por instancia, incluidos los giros semicirculares de cabecera, pero no
  para máscaras con ramificaciones en Y) y mide su longitud real en metros.
- **Pose** (`keypoints`, `kpt_shape: [2, 3]`): la línea es directamente el
  segmento recto entre los 2 keypoints que reporta el modelo — no hay curva que
  seguir, el propio dataset representa cada instancia como el tramo entre sus
  dos extremos visibles en esa foto/recorte (no la línea de siembra completa de
  punta a punta del cultivo).

Cada instancia se clasifica como sembrada o no sembrada por el **nombre de clase**
del modelo: si contiene alguna palabra de la lista en
`ai/inference/row_segmenter.py::_UNSOWN_KEYWORDS` (`no_sembr`, `falla`, `omitid`,
`vacio`/`vacío`, `gap`) se cuenta como falla; cualquier otro nombre (ej.
`linea_sembrada`, o `Crop_Row` del dataset de pose ya entrenado) se cuenta como
línea sembrada.

Un dataset de una sola clase también sirve — `_infer_gaps_between_rows` agrupa
las instancias sembradas por fila física (orientación y desplazamiento lateral
similares, dentro de la "separación aproximada entre surcos" configurada) y
reporta como posible falla cualquier hueco espacial entre detecciones
consecutivas más largo que la longitud mínima configurada. Es la vía que usa el
modelo de pose ya entrenado, que solo tiene la clase `Crop_Row`.

### Modelo ya entrenado: `row_lines_pose-2`

`scripts/train_row_lines_pose.py` entrenó un `yolo11m-pose` (kpt_shape `[2, 3]`)
sobre el dataset público "Crop-row" (Roboflow, 1000 imágenes). Resultado en
validación (126 épocas, `imgsz=640`, `batch=4`):

| | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Caja | 0.917 | 0.895 | 0.927 | 0.652 |
| Pose (keypoints) | 0.994 | 0.971 | 0.994 | 0.993 |

Es el modelo que trae precargado por defecto el plugin
(`config/default.yaml::ai.row_lines.weights_path`).

**Limitación importante de dominio**: este dataset son fotos a nivel de suelo
(cámara de robot/persona caminando entre las hileras), no vistas aéreas de dron.
La precisión de pose casi perfecta en validación refleja lo fácil que es ese
dataset específico (cada foto es un solo cultivo grande, keypoints casi siempre
en el borde superior/inferior del cuadro), no necesariamente que el modelo vaya
a generalizar bien sobre un ortomosaico real, muy distinto en escala y
perspectiva — mismo problema que ya se documentó para los modelos de detección
de plantas. Validá sus resultados sobre tu propio ortomosaico antes de confiar
en los metros calculados; si no generaliza, la heurística RGB (o un modelo
entrenado con recortes reales de tus ortomosaicos) sigue siendo la alternativa.

### Entrenar un modelo de segmentación (alternativa)

`scripts/train_row_lines.py` entrena `yolo11m-seg.pt` (segmentación de
instancias, no detección de cajas) sobre un dataset en
`data/images/roboflow_row_lines/` — todavía no se consiguió un dataset apto para
esto (ver `docs/BITACORA.md`). Convención de clases en `data.yaml` (así
`_UNSOWN_KEYWORDS` las clasifica bien):

```yaml
names: ['linea_sembrada', 'linea_no_sembrada']
```

Al etiquetar:

- Cada línea es UN polígono continuo que sigue su forma real, incluidas las
  curvas/semicírculos de cabecera donde la sembradora gira — no la partas en
  segmentos rectos ni la anotes como caja.
- No anotes el mismo tramo dos veces con clases distintas.
- Separa entrenamiento y validación por parcela/vuelo completo (no por recorte
  suelto), para no filtrar información del mismo surco entre splits.
- No uses las predicciones automáticas del modelo como verdad de referencia sin
  corregirlas a mano.

Al evaluar cualquier modelo entrenado (segmentación o pose), mirá además del mAP:
el error de posición del surco (¿la línea calculada queda centrada en la línea
real?) y la precisión/exhaustividad de las fallas detectadas — son las métricas
que de verdad importan para el cálculo de metros lineales.

`RowModelOptions` (en `ai/inference/row_segmenter.py`) agrupa los parámetros
propios del modelo (`weights_path`, `conf_threshold`, `iou_threshold`, `imgsz`);
los espaciales (`resolution_m`, `spacing_m`, `min_row_m`, `max_pixels`) se
reutilizan de `RowOptions`, la misma configuración que usa la heurística RGB.

## Verificación

`python -m pytest tests/test_row_detection.py tests/test_row_segmenter.py tests/test_vegetation_index.py tests/test_analysis.py -q`

Con el Python de QGIS: `python-qgis.bat tests/qgis_rows_smoke.py` comprueba tarea,
guardado, creación del diálogo y carga de capas de líneas con sus colores.
