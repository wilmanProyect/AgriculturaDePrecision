# Estado del proyecto — AgriculturaDePrecision

Foto del estado actual del sistema (2026-09-17). Para el detalle de cómo se
llegó hasta acá, ver `docs/BITACORA.md`. Para el detalle técnico de líneas de
siembra específicamente, ver `docs/lineas_siembra.md`.

## Arquitectura

```
core/           Motor GIS (independiente de QGIS y de la IA)
  gis/raster/     RasterManager, metadatos, utilidades (pixel<->coords)
  gis/vector/     VectorManager, metadatos, utilidades
  geometry/       GeometryManager, GeometryValidator
  analysis/       SpatialAnalysis (point-in-polygon, buffer, overlay)
                  VegetationIndexCalculator (NDVI/NDRE/ExG/VARI)
                  RowDetector (líneas de siembra, heurística RGB)
  exports/        ExportManager (CSV, etc.)

ai/             Motor IA (YOLO11 / Ultralytics)
  training/       TrainingConfig, YOLOTrainer (entrena cualquier variante YOLO11)
  inference/      PlantDetector (detección de plantas, cajas)
                  RowSegmenter (líneas de siembra, YOLO11-seg)

plugin_qgis/    Plugin de QGIS que conecta core + ai a la interfaz
  dialog.py       Diálogo principal (PrecisionAgDialog)
  worker.py       Tareas en segundo plano (QgsTask): DetectionTask,
                  VegetationIndexTask, RowAnalysisTask
  plugin.py       Punto de entrada (classFactory, menú/toolbar)

scripts/        Scripts de entrenamiento (fuera del plugin)
data/images/    Datasets (gitignored)
data/models/    Modelos entrenados y sus métricas (gitignored)
```

## Modelos entrenados

| Modelo | Arquitectura | Clases | mAP50 / mAP50-95 | Confiabilidad |
|---|---|---|---|---|
| `crop_weed` | YOLO11n | crop, weed | 0.818 / 0.582 | ⚠️ val de 8 imágenes, ruidoso |
| `seedling` | YOLO11n | Seedling | 0.905 / 0.529 | ✅ val de 142 imágenes, sólido |
| `weeding_detection` | YOLO11n | Plant | 0.854 / 0.470 | ⚠️ val de 8 imágenes, ruidoso |
| `row_lines_pose-2` | YOLO11m-pose | Crop_Row (kpt) | caja 0.652, pose 0.993 | ❌ descartado: aprendió a dibujar siempre una línea vertical (sesgo del dataset), ignora la orientación real de las hileras. Ver `docs/BITACORA.md` #13. |

Sin entrenar todavía: `plant_counting`, `weed_crop_large`. En preparación:
dataset propio de líneas de siembra con imágenes aéreas reales del usuario
(1077 fotos DJI, ver `docs/BITACORA.md` #14), etiquetándose en Roboflow con
segmentación de instancias (no pose, para poder representar curvas).

**Estándar del proyecto**: los entrenamientos nuevos usan `yolo11m.pt` (o su
variante `-seg`/`-pose`) como checkpoint base, no `yolo11n.pt`. Los 3 modelos ya
entrenados con `yolo11n` no se reentrenaron.

## Limitación conocida: dominio de las imágenes de entrenamiento

Los 4 datasets usados hasta ahora (`crop_weed`, `seedling`, `weeding_detection`,
`Crop-row.v12`) son **fotos terrestres** (celular/robot a ras de suelo o altura
de cintura), no vistas aéreas de dron. Un modelo entrenado con ese tipo de
imagen generaliza mal sobre un ortomosaico (distinta escala, perspectiva,
iluminación) — produce falsos positivos dispersos que no coinciden con la
vegetación real. **No es un bug de código**: se auditó toda la cadena de
georreferenciación (tiles → píxeles → coordenadas → CRS → point-in-polygon) y es
correcta.

Para que la detección funcione de forma confiable sobre ortomosaicos hace falta
entrenar (o afinar) con imágenes aéreas reales — idealmente recortes de los
propios ortomosaicos del usuario, etiquetados a mano.

## Funcionalidades del plugin QGIS

1. **Detectar y Contar Plantas** (`DetectionTask` / `PlantDetector`): YOLO11 por
   tiles sobre el ortomosaico, conteo por parcela, densidad y coloreado
   automático. Requiere un archivo de pesos `.pt` (cualquiera de los 3 modelos
   entrenados, elegido a mano en el diálogo).
2. **Índice de Vegetación** (`VegetationIndexTask` / `VegetationIndexCalculator`):
   NDVI/NDRE (requieren ráster de índice ya generado, sensor NIR/RedEdge) o
   ExG/VARI (calculados al vuelo desde RGB, sin sensor extra).
3. **Líneas de siembra y posibles fallas** (`RowAnalysisTask`), dos botones:
   - **"...(RGB, sin modelo)"**: heurística RGB (`RowDetector`), sin ningún
     campo de IA — el flujo recomendado por ahora, dado el punto anterior.
   - **"...(avanzado: RGB o modelo YOLO)"**: igual, pero con un campo opcional
     para indicar un modelo YOLO (segmentación o pose, `RowSegmenter`) — vacío
     por defecto hasta tener un modelo entrenado con imágenes aéreas propias.
   La resolución de análisis es automática por defecto (detecta la del
   ortomosaico real, acotada a un límite de memoria por parcela). Ambos métodos
   infieren huecos entre detecciones aunque el modelo/heurística no distinga
   clases de falla explícitas. Exporta GeoPackage + estilo QML + resumen JSON
   por corrida, y muestra en el log los metros sembrados y no sembrados.

## Cómo verificar / reproducir

```bash
# Toda la suite (112 tests al momento de escribir esto)
python -m pytest -q

# Solo líneas de siembra (heurística + YOLO-seg)
python -m pytest tests/test_row_detection.py tests/test_row_segmenter.py -q

# Smoke test dentro del entorno de QGIS (requiere python-qgis.bat)
python-qgis.bat tests/qgis_rows_smoke.py
```

Regenerar el zip del plugin para instalar en QGIS (Complementos → Instalar desde
ZIP): empaquetar `plugin_qgis/` (sin `__pycache__/`) en
`AgriculturaDePrecision-plugin.zip`. `core/` y `ai/` **no** van dentro del zip:
se instalan aparte (`pip install -e .` / `pip install --user`) y
`plugin_qgis/__init__.py` agrega el `site-packages` de usuario al `sys.path` de
QGIS para encontrarlos.

## Próximos pasos sugeridos

- Etiquetar en Roboflow el banco de imágenes aéreas propio (segmentación de
  instancias, prestando atención a curvas de cabecera y cruces) y entrenar con
  `scripts/train_row_lines.py` (ya con aumento de datos por rotación activado).
- Entrenar `plant_counting` y `weed_crop_large` si siguen siendo relevantes, y
  considerar reentrenar los 3 modelos de plantas con imágenes aéreas reales
  (mismo problema de dominio que afectó a `row_lines_pose-2`).
- Definir un mecanismo de respaldo para `data/models/` (no está en git): backup
  manual, Git LFS o DVC, para no perder los modelos entrenados.
- Verificar en QGIS real que el fix del crash nativo de PROJ (mover
  `estimate_utm_crs()`/`to_crs()` al hilo principal, ver `docs/BITACORA.md` #18)
  resuelve el problema de forma definitiva.
