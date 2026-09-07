# -*- coding: utf-8 -*-
"""
Detector de plantas basado en YOLO11 (Ultralytics).
Ejecuta inferencia sobre imágenes individuales o sobre ortomosaicos completos
mediante tiling, y georreferencia cada detección usando el Motor GIS.
"""

import os
from typing import Dict, List, Optional

import numpy as np
from rasterio.windows import Window

from core.exceptions import InferenceError, ModelLoadError, ModelNotFoundError
from core.gis.raster.raster_manager import RasterManager
from core.logger import get_logger, log_execution_time

from ai.inference.detection_result import Detection
from ai.utils.georeferencing import georeference_detections
from ai.utils.nms import non_max_suppression

logger = get_logger(__name__)

# Clases por defecto del modelo de detección de plantas (Fase 2 del proyecto)
DEFAULT_CLASS_NAMES = {0: "planta", 1: "arbol", 2: "espacio_vacio"}


class PlantDetector:
    """Clase responsable de cargar un modelo YOLO11 y ejecutar inferencia sobre imágenes u ortomosaicos."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640
    ):
        self.weights_path = weights_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Indica si el modelo YOLO11 ya fue cargado en memoria."""
        return self._model is not None

    def load_model(self) -> None:
        """
        Carga el modelo YOLO11 desde `weights_path`.
        Lanza ModelNotFoundError si el archivo de pesos no existe, o ModelLoadError
        si Ultralytics no puede cargarlo (formato inválido, checkpoint corrupto, etc.).
        """
        if not os.path.exists(self.weights_path):
            raise ModelNotFoundError(
                f"No se encontró el archivo de pesos del modelo: {self.weights_path}"
            )

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
            logger.info(f"Modelo YOLO11 cargado correctamente: {self.weights_path} (device={self.device})")
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelLoadError(f"No se pudo cargar el modelo '{self.weights_path}': {e}") from e

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Debe llamar a load_model() antes de ejecutar inferencia.")

    @log_execution_time(logger)
    def predict(self, image: np.ndarray) -> List[Detection]:
        """
        Ejecuta inferencia YOLO11 sobre una única imagen/tile (array HxWxC o HxW).
        Devuelve las detecciones en coordenadas de píxel de esa imagen (sin georreferenciar).
        """
        self._ensure_loaded()

        try:
            results = self._model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            raise InferenceError(f"Fallo durante la inferencia YOLO11: {e}") from e

        detections: List[Detection] = []
        for result in results:
            names = getattr(result, "names", DEFAULT_CLASS_NAMES)
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_pixel=(x1, y1, x2, y2),
                    center_pixel=((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                ))

        return detections

    @log_execution_time(logger)
    def predict_orthomosaic(
        self,
        raster_manager: RasterManager,
        tile_size: int = 1024,
        overlap: float = 0.2,
        bands: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        Ejecuta inferencia YOLO11 sobre un ortomosaico completo, dividiéndolo en tiles
        solapados para no perder detecciones en los bordes. Las coordenadas de cada
        detección se reescalan al espacio de píxel del ortomosaico completo, se fusionan
        los duplicados de los solapes con NMS, y finalmente se georreferencian usando
        la transformación afín del ráster.
        """
        self._ensure_loaded()

        if raster_manager.dataset is None:
            raise RuntimeError("El RasterManager debe tener un ortomosaico abierto (open()) antes de inferir.")

        dataset = raster_manager.dataset
        width, height = dataset.width, dataset.height
        band_indices = bands or list(range(1, min(dataset.count, 3) + 1))

        stride = max(1, int(tile_size * (1 - overlap)))
        all_detections: List[Detection] = []

        for row_off in range(0, height, stride):
            for col_off in range(0, width, stride):
                win_width = min(tile_size, width - col_off)
                win_height = min(tile_size, height - row_off)
                window = Window(col_off, row_off, win_width, win_height)

                tile = dataset.read(band_indices, window=window)
                tile_image = self._prepare_tile(tile)

                tile_detections = self.predict(tile_image)
                for detection in tile_detections:
                    x1, y1, x2, y2 = detection.bbox_pixel
                    detection.bbox_pixel = (x1 + col_off, y1 + row_off, x2 + col_off, y2 + row_off)
                    cx, cy = detection.center_pixel
                    detection.center_pixel = (cx + col_off, cy + row_off)

                all_detections.extend(tile_detections)

        merged = non_max_suppression(all_detections, iou_threshold=self.iou_threshold)
        georeference_detections(merged, tuple(dataset.transform))

        logger.info(
            f"Inferencia sobre ortomosaico finalizada: {len(all_detections)} detecciones en tiles, "
            f"{len(merged)} tras fusionar solapes."
        )
        return merged

    @staticmethod
    def _prepare_tile(tile: np.ndarray) -> np.ndarray:
        """
        Convierte un tile leído con rasterio (bandas, alto, ancho) al formato
        (alto, ancho, bandas) esperado por Ultralytics, replicando la banda
        si el ráster es de una sola banda (escala de grises).
        """
        image = np.moveaxis(tile, 0, -1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image

    @staticmethod
    def count_by_class(detections: List[Detection]) -> Dict[str, int]:
        """Devuelve el número de detecciones agrupadas por nombre de clase."""
        counts: Dict[str, int] = {}
        for detection in detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return counts
