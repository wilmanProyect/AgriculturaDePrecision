# -*- coding: utf-8 -*-
"""
Detección de líneas de siembra con un modelo YOLO11 entrenado por el usuario:
soporta tanto segmentación de instancias (`scripts/train_row_lines.py`,
máscaras/polígonos) como pose/keypoints (`scripts/train_row_lines_pose.py`,
caja + 2 extremos por línea — el formato del dataset "Crop-row" ya entrenado).
El tipo de modelo se detecta solo, por lo que expone el modelo en `result`
(`masks` vs `keypoints`); no hace falta indicarlo aparte.

A diferencia de `RowDetector` (heurística RGB en `core.analysis.row_detection`,
que sigue crestas de vegetación), aquí cada línea/falla es una instancia que el
modelo ya localiza directamente: no hace falta estimar orientación ni umbral de
vegetación. Devuelve el mismo formato que `RowDetector.analyze()`
(`rows_gdf` = líneas sembradas, `gaps_gdf` = líneas no sembradas) para poder
conectarse a la misma `RowAnalysisTask` sin cambiar cómo se guardan o pintan
los resultados.

Convención de clases esperada en el dataset/modelo (ver docs/lineas_siembra.md):
el nombre de clase que contenga alguna palabra de `_UNSOWN_KEYWORDS` (ej.
'linea_no_sembrada') se cuenta como falla; cualquier otro nombre (ej.
'linea_sembrada', o 'Crop_Row' del dataset de pose) se cuenta como surco
sembrado.

Un dataset de una sola clase (como el de pose ya entrenado) también sirve:
además de la clase explícita de falla, `_infer_gaps_between_rows` agrupa las
instancias sembradas por fila física y reporta como posible falla cualquier
hueco espacial entre detecciones consecutivas más largo que `RowOptions.min_row_m`.
"""
import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window, from_bounds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString
from skimage.morphology import skeletonize

from core.analysis.row_detection import (
    RowOptions, _check_cancel, _frame, resolve_resolution_m, split_large_geometries, tile_area_m2
)
from core.exceptions import InferenceError, ModelLoadError, ModelNotFoundError
from core.logger import get_logger, log_execution_time

logger = get_logger(__name__)

# Nombres de clase que se cuentan como línea NO sembrada / falla. Cualquier otro
# nombre de clase se trata como línea sembrada. Ajustar aquí si el dataset usa
# otra convención de nombres.
_UNSOWN_KEYWORDS = ("no_sembr", "no sembr", "falla", "omitid", "vacio", "vacío", "gap")


def _looks_unsown(class_name: str) -> bool:
    name = class_name.lower()
    return any(keyword in name for keyword in _UNSOWN_KEYWORDS)


def _line_endpoints(line: LineString) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(line.coords, dtype=float)
    return coords[0], coords[-1]


def _line_unit_direction(line: LineString) -> np.ndarray:
    p0, p1 = _line_endpoints(line)
    vector = p1 - p0
    norm = float(np.hypot(*vector))
    return vector / norm if norm > 1e-9 else np.array([1.0, 0.0])


def _unsigned_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Ángulo entre dos direcciones tratando la línea como no orientada (0-90°)."""
    cos_angle = abs(float(np.clip(np.dot(u, v), -1.0, 1.0)))
    return math.degrees(math.acos(cos_angle))


def _perpendicular_distance(point: np.ndarray, ref_point: np.ndarray, ref_direction: np.ndarray) -> float:
    normal = np.array([-ref_direction[1], ref_direction[0]])
    return abs(float(np.dot(point - ref_point, normal)))


def _infer_gaps_between_rows(sown_records: list, min_gap_m: float, max_lateral_offset_m: float) -> list:
    """
    Agrupa instancias 'sembrada' que forman la misma fila física (orientación y
    desplazamiento lateral similares, dentro de `max_lateral_offset_m`) y
    sintetiza un tramo de "posible falla" en cada hueco entre instancias
    consecutivas de esa fila más largo que `min_gap_m`. El hueco entre dos
    detecciones también es evidencia de falta de siembra, incluso si el modelo
    no tiene una clase explícita de falla (dataset de una sola clase).
    Aproximación por segmentos: usa el punto extremo de cada instancia en la
    dirección de avance de la fila, válido para instancias razonablemente rectas.
    """
    pending = list(sown_records)
    clusters: List[list] = []
    while pending:
        seed = pending.pop(0)
        seed_dir = _line_unit_direction(seed['geometry'])
        seed_p0, _ = _line_endpoints(seed['geometry'])
        cluster = [seed]
        rest = []
        for item in pending:
            direction = _line_unit_direction(item['geometry'])
            if _unsigned_angle_deg(seed_dir, direction) > 20:
                rest.append(item)
                continue
            midpoint = np.mean(np.asarray(item['geometry'].coords, dtype=float), axis=0)
            if _perpendicular_distance(midpoint, seed_p0, seed_dir) > max_lateral_offset_m:
                rest.append(item)
                continue
            cluster.append(item)
        pending = rest
        clusters.append((seed_dir, cluster))

    gaps = []
    for axis, cluster in clusters:
        if len(cluster) < 2:
            continue

        def projection(item):
            p0, _ = _line_endpoints(item['geometry'])
            return float(np.dot(p0, axis))

        ordered = sorted(cluster, key=projection)
        for prev_item, next_item in zip(ordered, ordered[1:]):
            prev_pts = np.asarray(prev_item['geometry'].coords, dtype=float)
            next_pts = np.asarray(next_item['geometry'].coords, dtype=float)
            end_point = prev_pts[np.argmax(prev_pts @ axis)]
            start_point = next_pts[np.argmin(next_pts @ axis)]
            gap_len = float(np.linalg.norm(end_point - start_point))
            if gap_len >= min_gap_m:
                gaps.append({'geometry': LineString([tuple(end_point), tuple(start_point)]), 'long_m': gap_len})
    return gaps


@dataclass(frozen=True)
class RowModelOptions:
    """Parámetros propios del modelo YOLO-seg (los espaciales viven en `RowOptions`)."""

    weights_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640  # debe rondar el imgsz de entrenamiento del modelo (640 para row_lines_pose)

    def validate(self):
        if not os.path.exists(self.weights_path):
            raise ModelNotFoundError(f"No se encontró el archivo de pesos: {self.weights_path}")
        if not 0 < self.conf_threshold <= 1 or not 0 < self.iou_threshold <= 1:
            raise ValueError("Confianza e IoU deben estar en (0, 1].")


def _crop_orthomosaic_rgb(vrt, bands, polygon, resolution_m: float, max_pixels: int):
    """
    Recorta y remuestrea a `resolution_m` la región RGB de `vrt` cubierta por `polygon`,
    devolviendo una imagen uint8 (H, W, 3) lista para YOLO, su máscara de píxeles válidos
    (dentro del polígono y sin NoData) y la transformación afín del recorte.
    Réplica simplificada de `RowDetector._crop` (sin el cálculo de ExG): se mantiene
    separada para no acoplar el método heurístico con el basado en modelo.
    """
    from rasterio.errors import WindowError
    try:
        requested = from_bounds(*polygon.bounds, transform=vrt.transform)
        left, top = math.floor(requested.col_off), math.floor(requested.row_off)
        window = Window(left, top, math.ceil(requested.col_off + requested.width) - left,
                        math.ceil(requested.row_off + requested.height) - top)
        window = window.intersection(Window(0, 0, vrt.width, vrt.height))
    except WindowError:
        return None
    bounds = rasterio.windows.bounds(window, vrt.transform)
    width = max(1, math.ceil((bounds[2] - bounds[0]) / resolution_m))
    height = max(1, math.ceil((bounds[3] - bounds[1]) / resolution_m))
    if width * height > max_pixels:
        raise ValueError("Parcela demasiado grande para esta resolución. Divide la parcela o aumenta la resolución en metros/píxel.")
    transform = vrt.window_transform(window) * Affine.scale(window.width / width, window.height / height)
    data = vrt.read(bands, window=window, out_shape=(3, height, width), masked=True,
                    out_dtype='float32', resampling=Resampling.bilinear)
    valid = ~np.ma.getmaskarray(data).any(axis=0)
    if ColorInterp.alpha in vrt.colorinterp:
        alpha = vrt.read(vrt.colorinterp.index(ColorInterp.alpha) + 1, window=window,
                         out_shape=(height, width), resampling=Resampling.nearest)
        valid &= alpha > 0
    valid &= geometry_mask([polygon], (height, width), transform, invert=True)
    if not valid.any():
        return None
    image = np.clip(data.filled(0), 0, 255).astype('uint8')
    image = np.moveaxis(image, 0, -1)  # (bands, H, W) -> (H, W, bands), formato esperado por Ultralytics
    return image, valid, transform


def _skeleton_line(mask: np.ndarray, transform: Affine, pixel_size_m: float, min_length_m: float
                    ) -> Optional[Tuple[LineString, float]]:
    """
    Reduce una máscara de instancia a su línea central (esqueleto) y mide su longitud
    real siguiendo el camino más largo del esqueleto (barrido doble sobre el grafo de
    píxeles conectados): funciona para curvas simples, incluidos los giros semicirculares
    de cabecera, siempre que la máscara no tenga ramificaciones en Y.
    """
    skeleton = skeletonize(mask)
    ys, xs = np.nonzero(skeleton)
    n = len(ys)
    if n < 2:
        return None

    index_of = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    rows_idx, cols_idx, weights = [], [], []
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = index_of.get((int(y) + dy, int(x) + dx))
                if j is not None:
                    rows_idx.append(i)
                    cols_idx.append(j)
                    weights.append(math.hypot(dy, dx))
    graph = coo_matrix((weights, (rows_idx, cols_idx)), shape=(n, n)).tocsr()

    # Barrido doble: el nodo más lejano de uno cualquiera es un extremo; el más lejano
    # de ese extremo es el otro. Aproxima bien el camino más largo en un esqueleto
    # sin ramificaciones significativas (una única línea/falla por instancia).
    dist0, _ = dijkstra(graph, indices=0, return_predecessors=True)
    dist0 = np.where(np.isinf(dist0), -1.0, dist0)
    a = int(np.argmax(dist0))
    dist_a, pred_a = dijkstra(graph, indices=a, return_predecessors=True)
    dist_a = np.where(np.isinf(dist_a), -1.0, dist_a)
    b = int(np.argmax(dist_a))
    length_px = float(dist_a[b])
    if length_px * pixel_size_m < min_length_m:
        return None

    path = [b]
    while path[-1] != a:
        prev = int(pred_a[path[-1]])
        if prev < 0:
            break
        path.append(prev)
    path.reverse()
    if len(path) < 2:
        return None

    coords = [transform * (xs[idx] + 0.5, ys[idx] + 0.5) for idx in path]
    return LineString(coords), length_px * pixel_size_m


def _keypoints_line(kpts_xy: np.ndarray, valid: np.ndarray, transform: Affine, min_length_m: float
                     ) -> Optional[Tuple[LineString, float]]:
    """
    Construye la línea entre los 2 extremos que reporta un modelo de pose
    (kpt_shape=[2,3], ej. `scripts/train_row_lines_pose.py`): a diferencia de una
    máscara, no hay curva que seguir — el propio dataset representa cada
    instancia como el tramo recto entre sus dos extremos visibles en esa
    foto/recorte, no la línea de siembra completa (ver docs/lineas_siembra.md).
    """
    if kpts_xy.shape[0] < 2 or not np.all(np.isfinite(kpts_xy[:2])):
        return None
    height, width = valid.shape
    any_inside = False
    for x, y in kpts_xy[:2]:
        row, col = int(round(float(y))), int(round(float(x)))
        if 0 <= row < height and 0 <= col < width and valid[row, col]:
            any_inside = True
            break
    if not any_inside:
        return None
    (x0, y0), (x1, y1) = kpts_xy[0], kpts_xy[1]
    world0 = transform * (float(x0) + 0.5, float(y0) + 0.5)
    world1 = transform * (float(x1) + 0.5, float(y1) + 0.5)
    length_m = math.dist(world0, world1)
    if length_m < min_length_m:
        return None
    return LineString([world0, world1]), length_m


class RowSegmenter:
    """
    Carga un modelo YOLO11 (segmentación o pose) y detecta líneas de siembra sobre
    recortes RGB de un ortomosaico. El tipo de instancia que devuelve `segment()`
    depende de la tarea del modelo cargado: máscara (`masks`) para segmentación,
    o par de keypoints (`keypoints`) para pose — `analyze()` maneja ambos casos.
    """

    def __init__(self, options: RowModelOptions):
        options.validate()
        self.options = options
        self._model = None

    def load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.options.weights_path)
            logger.info(
                f"Modelo de líneas de siembra cargado: {self.options.weights_path} "
                f"(task={getattr(self._model, 'task', '?')})"
            )
        except Exception as e:
            raise ModelLoadError(f"No se pudo cargar el modelo '{self.options.weights_path}': {e}") from e

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("Debe llamar a load_model() antes de detectar líneas.")

    def segment(self, image: np.ndarray) -> List[Tuple[int, str, float, Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Ejecuta el modelo sobre una imagen (H, W, 3) y devuelve, por instancia,
        `(class_id, class_name, confidence, mask, keypoints_xy)`: exactamente uno
        de `mask` (array booleano) / `keypoints_xy` (array (2, 2) en píxeles) viene
        poblado según la tarea del modelo; el otro queda en `None`.
        """
        self._ensure_loaded()
        try:
            results = self._model.predict(
                source=image, conf=self.options.conf_threshold, iou=self.options.iou_threshold,
                imgsz=self.options.imgsz, retina_masks=True, verbose=False
            )
        except Exception as e:
            raise InferenceError(f"Fallo durante la inferencia de líneas de siembra: {e}") from e

        instances = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            masks = getattr(result, "masks", None)
            keypoints = getattr(result, "keypoints", None)
            for idx in range(len(boxes)):
                class_id = int(boxes.cls[idx])
                confidence = float(boxes.conf[idx])
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                mask_array = masks.data[idx].cpu().numpy().astype(bool) if masks is not None else None
                kpts_xy = keypoints.xy[idx].cpu().numpy() if keypoints is not None else None
                instances.append((class_id, class_name, confidence, mask_array, kpts_xy))
        return instances

    @log_execution_time(logger)
    def analyze(
        self,
        raster_path: str,
        parcelas_gdf: "gpd.GeoDataFrame",
        row_options: RowOptions = None,
        progress: Optional[Callable[[float], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        crs=None
    ) -> dict:
        """
        Segmenta líneas de siembra/fallas por parcela. `row_options` reutiliza los
        parámetros espaciales del método heurístico (resolución, longitud mínima,
        límite de píxeles) para que ambos métodos compartan la misma configuración
        y el mismo formato de resultado en `RowAnalysisTask`.

        `crs` es opcional: si el llamador ya reprojectó `parcelas_gdf` a un CRS
        métrico (recomendado desde un hilo en segundo plano de QGIS), pasarlo evita
        volver a llamar `estimate_utm_crs()`/`to_crs()` acá — ver la nota en
        `core.analysis.row_detection.RowDetector.analyze` sobre el crash nativo de
        PROJ que puede evitar.
        """
        if parcelas_gdf.empty or parcelas_gdf.crs is None:
            raise ValueError("Selecciona parcelas con geometría y CRS definidos.")
        self._ensure_loaded()
        row_options = row_options or RowOptions()

        if crs is not None:
            metric = parcelas_gdf
        else:
            crs = parcelas_gdf.estimate_utm_crs()
            if crs is None:
                raise ValueError("No se pudo determinar un CRS métrico para las parcelas.")
            metric = parcelas_gdf.to_crs(crs)
        row_options = resolve_resolution_m(row_options, raster_path, metric)
        for i, geom in enumerate(metric.geometry):
            if geom is None or geom.is_empty or geom.geom_type not in ('Polygon', 'MultiPolygon'):
                raise ValueError(f"Parcela {i + 1}: geometría poligonal vacía o inválida.")
            if not geom.is_valid:
                raise ValueError(f"Parcela {i + 1}: repara su geometría antes de analizar.")

        # Lotes grandes (ej. 150 ha) se dividen en piezas que respeten max_pixels a
        # la resolución elegida, para no necesitar un único recorte gigante en
        # memoria. La inferencia de huecos entre detecciones (más abajo) queda
        # acotada a cada pieza: un hueco justo en el límite entre piezas vecinas
        # no se detecta — no afecta la suma total de metros sembrados.
        resolution_m = row_options.resolution_m
        pieces = split_large_geometries(metric, tile_area_m2(row_options))

        rows, gaps, warnings = [], [], []

        with rasterio.open(raster_path) as src:
            if src.crs is None or src.count < 3:
                raise ValueError("Se requiere un ortomosaico RGB con CRS definido.")
            colors = src.colorinterp
            rgb = [colors.index(c) + 1 for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)] if all(
                c in colors for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)) else [1, 2, 3]
            grid, grid_width, grid_height = calculate_default_transform(
                src.crs, crs, src.width, src.height, *src.bounds, resolution=resolution_m)
            with WarpedVRT(src, crs=crs, transform=grid, width=grid_width, height=grid_height,
                           resampling=Resampling.bilinear, add_alpha=ColorInterp.alpha not in colors) as vrt:
                total = len(pieces)
                for idx, (label, geom) in enumerate(pieces, start=1):
                    _check_cancel(should_stop)
                    if geom is None or geom.is_empty or geom.geom_type not in ('Polygon', 'MultiPolygon'):
                        continue  # pieza degenerada del recorte en grilla, no un error del usuario
                    crop = _crop_orthomosaic_rgb(vrt, rgb, geom, resolution_m, row_options.max_pixels)
                    if crop is None:
                        warnings.append(f"Parcela {label}: sin cobertura RGB válida.")
                        continue
                    image, valid, transform = crop
                    instances = self.segment(image)
                    if not instances:
                        warnings.append(f"Parcela {label}: el modelo no detectó líneas.")
                    parcela_rows = []
                    for class_id, class_name, confidence, mask, kpts_xy in instances:
                        _check_cancel(should_stop)
                        if mask is not None:
                            mask = mask & valid
                            if not mask.any():
                                continue
                            traced = _skeleton_line(mask, transform, resolution_m, row_options.min_row_m)
                        elif kpts_xy is not None:
                            traced = _keypoints_line(kpts_xy, valid, transform, row_options.min_row_m)
                        else:
                            continue
                        if traced is None:
                            continue
                        line, length_m = traced
                        record = dict(parcela_id=label, surco_id=0, sector=1, long_m=length_m,
                                      estado=f"{class_name} ({confidence:.2f})", geometry=line)
                        if _looks_unsown(class_name):
                            record['umbral_m'] = row_options.min_row_m
                            gaps.append(record)
                        else:
                            parcela_rows.append(record)
                    rows.extend(parcela_rows)
                    # Huecos espaciales entre instancias sembradas de una misma fila: evidencia de
                    # falta de siembra incluso sin una clase explícita de falla en el modelo.
                    for gap in _infer_gaps_between_rows(parcela_rows, row_options.min_row_m, row_options.spacing_m):
                        gaps.append(dict(parcela_id=label, surco_id=0, sector=1, long_m=gap['long_m'],
                                          estado='Posible falla (hueco entre detecciones)',
                                          umbral_m=row_options.min_row_m, geometry=gap['geometry']))
                    if progress:
                        progress(100 * idx / total)
        _check_cancel(should_stop)
        for offset, record in enumerate(rows, start=1):
            record['surco_id'] = offset
        for offset, record in enumerate(gaps, start=1):
            record['surco_id'] = offset
        method = f"YOLO11-{getattr(self._model, 'task', 'seg')}"
        return {'rows_gdf': _frame(rows, crs), 'gaps_gdf': _frame(gaps, crs, True),
                'warnings': warnings, 'method': method, 'options': row_options}
