# Bitácora del proyecto — AgriculturaDePrecision

Registro cronológico de avances, problemas encontrados y cómo se resolvieron.
Complementa (no reemplaza) `docs/ESTADO_DEL_PROYECTO.md` (foto del estado
actual) y `docs/lineas_siembra.md` (documentación técnica de esa función).

---

## 2026-09-08 a 2026-09-10 — Primeros 3 entrenamientos YOLO11n

Se entrenaron 3 de los 5 datasets de `data/images/` con `scripts/train_datasets.py`
(YOLO11n, 100 épocas, imgsz 640, autobatch, `patience=50`):

| Dataset | Clases | Imágenes train/val | Resultado (`best.pt`) |
|---|---|---|---|
| `crop_weed` | crop, weed | 84 / 8 | mAP50=0.818, mAP50-95=0.582 |
| `seedling` | Seedling | 1500 / 142 | mAP50=0.905, mAP50-95=0.529 |
| `weeding_detection` | Plant | 70 / 8 | mAP50=0.854, mAP50-95=0.470 |

Quedaron sin entrenar: `plant_counting`, `weed_crop_large`.

---

## 2026-09-11 — Sesión de revisión y expansión

### 1. Verificación de los 3 entrenamientos

Se recargó cada `best.pt` y se corrió `model.val()` contra su propio split de
validación para confirmar que Ultralytics había guardado el checkpoint de mejor
fitness (no el de la última época). Los tres reprodujeron exactamente las
métricas esperadas — el guardado de checkpoints funcionó bien.

**Problema detectado:** `crop_weed` y `weeding_detection` tienen solo 8 imágenes
de validación (y 84/70 de entrenamiento) → métricas estadísticamente poco
confiables, alto ruido época a época. `seedling` (142 imágenes val) es el más
sólido de los tres. No se resolvió (es una limitación del dataset, no del
código) — queda como advertencia permanente sobre la confiabilidad de esos dos
modelos.

### 2. Índices de vegetación ExG/VARI

Se encontró ya implementado en disco (sin commitear): `core/analysis/vegetation_index.py`
sumó `compute_exg`/`compute_vari` y `zonal_statistics_rgb_index`, calculables
directo desde bandas RGB (sin necesitar sensor NIR/RedEdge). Verificado con
`tests/test_vegetation_index.py` (20/20 tests).

### 3. Corrección de `config/default.yaml`

El `weights_path` por defecto apuntaba a `data/models/plant_detector/weights/best.pt`,
una ruta que nunca existió (se entrenó `crop_weed`/`seedling`/`weeding_detection`,
no `plant_detector`). **Resuelto**: se actualizó a `weeding_detection/weights/best.pt`
(y luego a `weeding_detection` tras confirmarlo con el usuario), y se corrigió
`architecture` para reflejar el checkpoint base real usado.

### 4. Bug reportado: detecciones "fuera de lugar" en QGIS

El usuario compartió una captura donde los puntos de "Plantas detectadas" no
caían sobre el cultivo real, y la capa de densidad se pintaba encima del
ortomosaico tapándolo.

**Diagnóstico:**
- Se auditó toda la cadena de georreferenciación (`pixel_to_coords`, offsets de
  tiles, reproyección de CRS, `point_in_polygon`) línea por línea — matemáticamente
  correcta, no había bug de coordenadas.
- La causa real: las imágenes de entrenamiento de los 3 modelos son fotos
  terrestres de celular en primer plano (plántulas individuales), no vistas
  aéreas de dron. El modelo `weeding_detection` nunca vio ese tipo de imagen y
  generaba falsos positivos sobre tierra/caminos al aplicarlo al ortomosaico.
  **No es un bug de código — es un problema de dominio de datos**, documentado
  como limitación (ver `docs/ESTADO_DEL_PROYECTO.md`).
- Sí había un problema real de UI: la capa "Parcelas - análisis" se pintaba a
  opacidad 100%, tapando el ortomosaico. **Resuelto**: `layer.setOpacity(0.55)`
  en `plugin_qgis/dialog.py::_apply_categorized_renderer`.

### 5. Cambio de estándar: YOLO11n → YOLO11m

A pedido del usuario ("ya no utilizaremos yolo11n"), se actualizó
`scripts/train_datasets.py`, `config/default.yaml` y `tests/test_ai_trainer.py`
para usar `yolo11m.pt` como checkpoint base de ahí en adelante. Los 3 modelos ya
entrenados siguen siendo YOLO11n (no se reentrenaron).

### 6. Descubrimiento: función de líneas de siembra (heurística RGB)

Apareció en disco (trabajo del usuario, no de esta sesión) una función nueva
completa: `core/analysis/row_detection.py` (`RowDetector`), que detecta surcos y
posibles fallas directamente de las bandas RGB del ortomosaico (ExG + estructura
tensorial + seguimiento de crestas), sin necesitar un modelo entrenado. Conectada
al plugin vía `RowAnalysisTask` (`plugin_qgis/worker.py`) y un botón nuevo en
`plugin_qgis/dialog.py`. Documentada en `docs/lineas_siembra.md`, con 6 tests en
`tests/test_row_detection.py`. Se verificó que los 94 tests del proyecto pasaban
en ese momento.

### 7. Pedido: integrar YOLO al análisis de líneas de siembra

El usuario pidió entrenar su propio modelo YOLO para detectar líneas sembradas,
líneas omitidas y calcular metros lineales sembrados/no sembrados (incluyendo
curvas semicirculares de cabecera, donde la sembradora gira).

**Implementado:**
- `ai/inference/row_segmenter.py` (`RowSegmenter`): corre un modelo YOLO11-seg
  sobre recortes georreferenciados por parcela, reduce cada máscara de instancia
  a su línea central real (esqueleto vía `skimage.morphology.skeletonize` +
  camino más largo del esqueleto vía `scipy.sparse.csgraph.dijkstra`, "barrido
  doble"), y mide su longitud en metros — funciona con curvas, no solo líneas
  rectas.
- `scripts/train_row_lines.py`: entrena `yolo11m-seg.pt` sobre un dataset con
  polígonos (no cajas), con convención de clases `linea_sembrada`/`linea_no_sembrada`.
  Devuelve el mismo formato (`rows_gdf`/`gaps_gdf`) que la heurística RGB, así que
  se conecta a `RowAnalysisTask` sin tocar el guardado de resultados.
- `plugin_qgis/dialog.py`: campo opcional "Modelo YOLO-seg (.pt)" en el
  sub-diálogo de líneas de siembra — vacío usa la heurística, con modelo usa YOLO.
- El log del plugin ahora muestra ambos totales (metros sembrados y no
  sembrados), antes solo mostraba las fallas.
- 5 tests nuevos en `tests/test_row_segmenter.py`. Total del proyecto: 99/99.

### 8. Recomendación de formato de dataset

Se aconsejó exportar de Roboflow en formato **YOLOv8** (no COCO JSON ni "YOLOv5
PyTorch", que suele exportar solo cajas) para preservar los polígonos.

### 9. Dataset "Crop-row.v12" — investigado, resultó no apto tal cual

El usuario encontró y descargó `Crop-row.v12-crop_row_1000.yolov8.zip`
(`Downloads/DatosYolloV11/`). Antes de entrenar con él se inspeccionó:

- **No es un dataset de segmentación**: `data.yaml` trae `kpt_shape: [2, 3]` —
  es un dataset de **pose/keypoints** (caja + 2 puntos = extremos de la línea en
  esa foto), no polígonos. Con 2 puntos no se puede representar una curva.
- **Una sola clase** (`Crop_Row`) — no distingue sembrado de no sembrado.
- **Fotos a nivel de suelo** (cámara de robot/persona caminando entre hileras),
  no vistas aéreas — mismo problema de dominio que el punto 4.

Se preguntó al usuario cómo seguir. Respuesta: el objetivo final sigue siendo el
ortomosaico de dron, y prefiere inferir huecos a partir del espaciado entre
detecciones en vez de depender de una segunda clase.

**Implementado en base a esa decisión:**
- Se extrajo el dataset a `data/images/roboflow_row_lines_pose/` y se corrigió
  su `data.yaml` (ruta absoluta, igual que los demás datasets).
- `scripts/train_row_lines_pose.py`: entrena `yolo11m-pose.pt` sobre este
  dataset (con aviso explícito de la limitación de dominio en el docstring).
- `ai/inference/row_segmenter.py::_infer_gaps_between_rows`: agrupa instancias
  "sembradas" por fila física (orientación + desplazamiento lateral similares) y
  sintetiza un tramo de "posible falla" en cada hueco entre detecciones
  consecutivas más largo que `min_row_m` — funciona aunque el modelo tenga una
  sola clase. 2 tests nuevos. Total del proyecto: 101/101.

### 10. Primer intento de entrenamiento del modelo pose — CUDA OOM

El usuario corrió `scripts/train_row_lines_pose.py` (yolo11m-pose, imgsz=960,
autobatch). Falló con `CUDA error: out of memory` en la RTX 2050 (4 GB VRAM): el
propio sondeo de AutoBatch se quedó sin memoria y devolvió un batch imposible
("584% de la GPU"), y el entrenamiento real falló después incluso con batch
reducido a 8.

**Resuelto**: se bajó `imgsz` a 640 y se fijó `batch=4` manual (se evita el
sondeo de AutoBatch, que fue el que falló). Se dejó documentado en el script un
plan B (`batch=2`, o `yolo11s-pose.pt`) por si volvía a faltar memoria.

### 11. Segundo intento — entrenando correctamente

Confirmado en curso al momento de escribir esta bitácora
(`data/models/row_lines_pose-2/`, época 13/150, sin errores). El fix de
memoria funcionó.

### 12. Segundo dataset de líneas de siembra — descarga bloqueada por Cloudflare

El usuario compartió un link de descarga directa de Roboflow Universe
(`universe.roboflow.com/ds/...`). El intento de descarga por `curl` devolvió una
página de verificación anti-bot de Cloudflare ("Just a moment...") en vez del
zip — Roboflow protege esos links y requiere un navegador real (JavaScript +
cookies). **Pendiente**: el usuario debe descargarlo manualmente desde el
navegador y pasar la ubicación del archivo para continuar.

### 13. `row_lines_pose-2` terminado — y confirmado que no sirve para el caso real

Entrenamiento completo (126/150 épocas): mAP50-95 caja=0.652, pose=0.993. La
cifra de pose casi perfecta hizo sospechar sesgo, no generalización real.

El usuario compartió una captura (`prueba_screen/Lineas de siembra YOLO.png`)
de una corrida sobre un ortomosaico real: la línea detectada no seguía ninguna
hilera visible. Se reprodujo el caso exacto con sus archivos reales
(`result.tif` + `poligono3.shp`) y se armó un overlay de las detecciones
crudas (`conf` bajado a 0.05): **las 4 detecciones eran casi verticales,
ignorando que las hileras reales del campo corren en diagonal**.

**Causa raíz identificada**: el dataset "Crop-row" siempre encuadraba la foto
igual (persona/robot caminando en el surco, hilera de arriba a abajo del
cuadro) → el modelo no aprendió "encontrar la hilera", aprendió el atajo
"dibujar una línea vertical de punta a punta". Sesgo estructural del dataset,
no un problema de calibración de umbrales — no tiene arreglo sin reentrenar
con orientaciones variadas. Se descartó seguir usando este modelo.

### 14. Banco de imágenes aéreas reales del usuario — sí es apto

El usuario consiguió 1077 fotos crudas de dron (DJI Mavic 3 Multispectral,
3 vuelos: A13/A7P-A7N/A8P-A8N, `Downloads/Imagenes para entrenamiento/`).
Verificado por EXIF/XMP: nadir real (gimbal -90°), ~85 m de altura,
~2,3 cm/píxel de resolución nativa — dominio correcto por primera vez.

Se creó `scripts/prepare_row_lines_dataset.py`: re-muestrea cada foto a la
resolución objetivo (evita el mismo desfase de escala del modelo de pose) y
sub-muestrea el solape entre fotos consecutivas de un mismo vuelo (típico
70-85%, etiquetar las 1077 sería redundante y arriesga fuga train/valid).

### 15. Resolución de análisis automática (antes fija en 0,05 m/píxel)

A pedido del usuario, `RowOptions.resolution_m` ahora acepta `None` (nuevo
default) = detectar la resolución nativa real del ortomosaico
(`native_resolution_m`), acotada para no superar `max_pixels` en la parcela
más grande (`resolve_resolution_m`). Probado contra `result.tif` real: detectó
2,66 cm/píxel solo, sin que nadie tuviera que adivinar un valor. UI: checkbox
"Detectar automáticamente" en el diálogo, marcado por defecto.

*(Nota: el achicado de resolución según el área quedó superado por la división
en piezas del punto 19 — ver ahí.)*

### 16. Sesgo de orientación en el dataset propio — mitigado con aumento de datos

El usuario notó que sus fotos crudas muestran casi siempre la hilera "hacia
arriba", pero en el ortomosaico las hileras aparecen diagonales/curvas/con
cruces — mismo riesgo que causó la falla del punto 13. Para segmentación
(polígonos, a diferencia de pose) esto sí tiene arreglo: se activó rotación
aleatoria completa (`degrees=180`) y espejado vertical (`flipud=0.5`) en
`scripts/train_row_lines.py` (`ai/training/trainer.py` ahora expone estos
hiperparámetros). Esto cubre la ORIENTACIÓN; las curvas de cabecera y cruces
siguen necesitando ejemplos reales etiquetados (no los inventa el aumento de
datos) — advertido al usuario para cuando elija qué recortes etiquetar.

Se ofreció generar pre-etiquetas con la heurística RGB para acelerar el
etiquetado; el usuario prefirió etiquetar desde cero en Roboflow.

### 17. Botón nuevo: heurística RGB sin ningún campo de IA

El usuario pidió volver a tener un botón que corra "el algoritmo de antes de
la IA" sin exponer el campo de modelo YOLO. Se agregó
**"Detectar líneas de siembra y posibles fallas (RGB, sin modelo)"**, separado
del botón avanzado (que sigue soportando YOLO opcional). De paso se sacó el
default de `config/default.yaml::ai.row_lines.weights_path` (apuntaba a
`row_lines_pose-2`, el modelo descartado en el punto 13) para que no se
precargue por accidente en el botón avanzado.

### 18. Crash nativo de PROJ al correr la heurística dentro de QGIS real

Primera corrida real del botón nuevo dentro de QGIS 4.2.2: **access violation**
(crash del proceso) en `pyproj`, dentro de `GeoDataFrame.estimate_utm_crs()`,
llamado desde `RowDetector.analyze()` corriendo en el hilo en segundo plano de
la `QgsTask`. Traza: `pyproj/crs/crs.py` → construcción del objeto CRS nativo
(`_CRS`) → `is_geographic` → `estimate_utm_crs`.

**Diagnóstico**: pyproj cachea el handle nativo de CRS por hilo
(`threading.local`); la primera vez que se toca desde un hilo nuevo (el de la
`QgsTask`) reconstruye ese handle llamando a PROJ — en QGIS 4.2.2 esa
reconstrucción cruzando hilos parece disparar el crash. No es un bug en la
lógica de detección: es un problema nativo de pyproj/PROJ/threading en ese
entorno específico, no capturable con `try/except` de Python.

**Resuelto**: se movió el cálculo de `estimate_utm_crs()` + `to_crs()` al hilo
principal (`plugin_qgis/dialog.py`, antes de crear la `QgsTask`), y se agregó
un parámetro `crs` opcional a `RowDetector.analyze()` y `RowSegmenter.analyze()`
para recibir el CRS ya resuelto y saltarse ese cálculo dentro del hilo en
segundo plano. Sin `crs` (uso fuera de QGIS, tests) el comportamiento es igual
que antes. 2 tests nuevos verifican que `estimate_utm_crs` no se llama cuando
se pasa `crs=`. Total del proyecto: 112/112.

### 19. Sigue fallando en lotes grandes — no era (solo) el crash de PROJ

El usuario reportó que seguía fallando y sospechó del tamaño del lote ("supero
las 10 ha"), pidiendo soportar hasta 150 ha (tiene lotes de ese tamaño).

**Causa real**: `RowDetector`/`RowSegmenter` recortaban la parcela COMPLETA
como un único array en memoria (acotado antes por `max_pixels`, coarseando la
resolución si hacía falta — ver punto 15). Para 150 ha a una resolución
utilizable (no 30+ cm/píxel, inservible para ver hileras), el recorte es
sencillamente demasiado grande para tenerlo entero en memoria — de ahí el
crash, aparte del de PROJ.

**Resuelto**: `resolve_resolution_m` ya no achica la resolución según el área
(ese mecanismo quedó eliminado); en cambio, `split_large_geometries` (nueva,
en `core/analysis/row_detection.py`) divide cualquier parcela cuya área supere
el presupuesto de píxeles (`max_pixels`, ahora interpretado POR PIEZA, no por
parcela completa) en una grilla de piezas más chicas, cada una intersecada con
la forma real del lote. `RowDetector.analyze()` y `RowSegmenter.analyze()`
procesan esa lista de piezas en vez de las parcelas originales — sin más
cambios en el resto del pipeline. Cada pieza mantiene trazabilidad al lote
original en su `parcela_id` ("N.K" para la pieza K del lote N).

Limitación aceptada: una hilera que cruza el límite entre dos piezas vecinas
queda partida en el resultado (más líneas, más cortas), y un hueco justo en
ese límite puede no detectarse — no afecta la suma total de metros reportada,
que es la cifra que realmente importa. Se agregó un margen del 15% al tamaño
de pieza para absorber el redondeo del recorte a límites de píxel (sin margen,
piezas muy ajustadas al límite podían disparar el mismo error "demasiado
grande" que se estaba tratando de eliminar). 6 tests nuevos (división de
geometrías, conservación de área, corrida completa con tiling en ambos
métodos). Total del proyecto: 116/116.

---

## Pendientes abiertos

- Verificar en QGIS real que los fixes del crash de PROJ (punto 18) y del
  manejo de lotes grandes (punto 19) resuelven el problema de forma
  definitiva sobre un lote real de ~150 ha — no se pudo reproducir ninguno de
  los dos crashes fuera de QGIS para confirmarlo 100%, y el tiempo de
  procesamiento de un lote de ese tamaño (muchas piezas) todavía no se midió
  en la práctica.
- Etiquetado en Roboflow del banco de imágenes aéreas (punto 14), con polígonos
  y prestando atención a curvas/cruces (punto 16).
- Conseguir/descargar el segundo dataset de líneas de siembra (bloqueado por
  Cloudflare desde el punto 12, pendiente de descarga manual) — puede que ya no
  haga falta si el dataset propio (punto 14) alcanza.
- `plant_counting` y `weed_crop_large` siguen sin entrenar.
- Ningún archivo de `data/images/*` ni `data/models/*` está en git (ver
  `.gitignore` actualizado) — no hay respaldo de los modelos entrenados fuera
  del disco local.
