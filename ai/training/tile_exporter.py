# -*- coding: utf-8 -*-
"""
Exporta tiles de un ortomosaico como imágenes individuales (JPG), listas para
anotar en herramientas como Roboflow, LabelImg o CVAT antes de entrenar YOLO11.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from rasterio.windows import Window, from_bounds

from core.gis.raster.raster_manager import RasterManager
from core.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class TileExporter:
    """Clase responsable de trocear un ortomosaico en imágenes individuales para anotación manual."""

    def __init__(self, tile_size: int = 640, overlap: float = 0.0, min_valid_fraction: float = 0.5):
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_valid_fraction = min_valid_fraction

    @log_execution_time(logger)
    def export_tiles(
        self,
        raster_manager: RasterManager,
        output_dir: str,
        prefix: str = "tile",
        bands: Optional[List[int]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> List[str]:
        """
        Recorre el ortomosaico (o la región indicada por `bounds`, en el CRS del
        ráster) en tiles y guarda cada uno como imagen JPG en `output_dir`.
        Descarta tiles mayormente vacíos (NoData) para no malgastar tiempo de
        anotación en zonas sin datos (bordes del vuelo, nubes, etc.).
        Devuelve la lista de rutas de archivo escritas.
        """
        if raster_manager.dataset is None:
            raise RuntimeError("El RasterManager debe tener un ortomosaico abierto (open()) antes de exportar tiles.")

        dataset = raster_manager.dataset
        band_indices = bands or list(range(1, min(dataset.count, 3) + 1))

        if bounds is not None:
            roi_window = from_bounds(*bounds, transform=dataset.transform)
            roi_window = roi_window.intersection(Window(0, 0, dataset.width, dataset.height))
            roi_col_off = max(0, int(roi_window.col_off))
            roi_row_off = max(0, int(roi_window.row_off))
            width = roi_col_off + int(roi_window.width)
            height = roi_row_off + int(roi_window.height)
        else:
            roi_col_off, roi_row_off = 0, 0
            width, height = dataset.width, dataset.height

        os.makedirs(output_dir, exist_ok=True)
        stride = max(1, int(self.tile_size * (1 - self.overlap)))
        written_paths: List[str] = []
        nodata = dataset.nodata

        for row_off in range(roi_row_off, height, stride):
            for col_off in range(roi_col_off, width, stride):
                win_width = min(self.tile_size, width - col_off)
                win_height = min(self.tile_size, height - row_off)
                window = Window(col_off, row_off, win_width, win_height)

                tile = dataset.read(band_indices, window=window)

                if nodata is not None:
                    valid_fraction = float(np.mean(tile[0] != nodata))
                    if valid_fraction < self.min_valid_fraction:
                        continue

                image = np.moveaxis(tile, 0, -1)
                if image.shape[-1] == 1:
                    image = np.repeat(image, 3, axis=-1)

                if image.dtype != np.uint8:
                    image = self._to_uint8(image)

                file_path = os.path.join(output_dir, f"{prefix}_{row_off}_{col_off}.jpg")
                Image.fromarray(image).save(file_path, quality=95)
                written_paths.append(file_path)

        logger.info(f"{len(written_paths)} tiles exportados a: {output_dir}")
        return written_paths

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """Escala una imagen a rango 0-255 (uint8) usando percentiles 2-98 para evitar outliers."""
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros_like(image, dtype=np.uint8)

        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            return np.zeros_like(image, dtype=np.uint8)

        scaled = np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0) * 255
        return scaled.astype(np.uint8)
