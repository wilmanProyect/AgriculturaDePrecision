# -*- coding: utf-8 -*-
"""
Módulo para administrar operaciones con archivos ráster (GeoTIFF).
"""

import os
import numpy as np
import rasterio
from typing import Optional, Dict

from core.logger import get_logger, log_execution_time
from core.exceptions import RasterNotFoundError, UnsupportedFormatError
from .raster_metadata import RasterMetadata

logger = get_logger(__name__)

class RasterManager:
    """Clase responsable de abrir, leer y analizar archivos ráster (GeoTIFF)."""

    def __init__(self):
        self.file_path: Optional[str] = None
        self.dataset: Optional[rasterio.DatasetReader] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @log_execution_time(logger)
    def open(self, file_path: str) -> None:
        """
        Abre un archivo ráster utilizando rasterio.
        """
        if not os.path.exists(file_path):
            raise RasterNotFoundError(f"El archivo ráster no existe en la ruta: {file_path}")

        # Validar extensión del archivo
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.tif', '.tiff', '.geotiff']:
            raise UnsupportedFormatError(f"Formato ráster no soportado: '{ext}'. Se requiere GeoTIFF (.tif/.tiff).")

        try:
            self.file_path = file_path
            self.dataset = rasterio.open(file_path)
            logger.info(f"Archivo ráster abierto correctamente: {file_path}")
        except Exception as e:
            logger.error(f"Error al abrir el ráster: {e}")
            raise

    def close(self) -> None:
        """Cierra el archivo ráster si está abierto."""
        if self.dataset:
            self.dataset.close()
            logger.info(f"Archivo ráster cerrado: {self.file_path}")
            self.dataset = None
            self.file_path = None

    def get_metadata(self) -> RasterMetadata:
        """
        Extrae y devuelve los metadatos estructurados del ráster.
        """
        if not self.dataset:
            raise RuntimeError("Debe abrir un archivo ráster antes de solicitar metadatos.")

        stat = os.stat(self.file_path)
        crs_str = self.dataset.crs.to_string() if self.dataset.crs else "Sin CRS definido"
        
        # Obtener resolución (tamaño de píxel)
        res_x = self.dataset.res[0]
        res_y = self.dataset.res[1]

        # Bounding box
        bounds_tuple = (
            self.dataset.bounds.left,
            self.dataset.bounds.bottom,
            self.dataset.bounds.right,
            self.dataset.bounds.top
        )

        return RasterMetadata(
            name=os.path.basename(self.file_path),
            path=os.path.abspath(self.file_path),
            width=self.dataset.width,
            height=self.dataset.height,
            count=self.dataset.count,
            crs=crs_str,
            res_x=res_x,
            res_y=res_y,
            transform=tuple(self.dataset.transform),
            bounds=bounds_tuple,
            dtype=str(self.dataset.dtypes[0]) if self.dataset.dtypes else "unknown",
            nodata=self.dataset.nodata,
            size_bytes=stat.st_size
        )

    def read_band(self, band_idx: int) -> np.ndarray:
        """
        Lee y devuelve los datos de la banda especificada (1-indexed).
        """
        if not self.dataset:
            raise RuntimeError("Debe abrir un archivo ráster antes de leer bandas.")

        if band_idx < 1 or band_idx > self.dataset.count:
            raise ValueError(f"Índice de banda '{band_idx}' fuera de rango. El ráster tiene {self.dataset.count} banda(s).")

        return self.dataset.read(band_idx)

    def get_statistics(self, band_idx: int) -> Dict[str, float]:
        """
        Calcula estadísticas (mínimo, máximo, media, desviación estándar, mediana)
        de la banda indicada, ignorando los valores de NoData.
        """
        band_data = self.read_band(band_idx)
        nodata_val = self.dataset.nodata

        # Filtrar valores NoData si existen
        if nodata_val is not None:
            valid_data = band_data[band_data != nodata_val]
        else:
            valid_data = band_data

        # Si no hay datos válidos tras el filtrado
        if valid_data.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0
            }

        return {
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "median": float(np.median(valid_data))
        }
