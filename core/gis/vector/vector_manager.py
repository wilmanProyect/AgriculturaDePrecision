# -*- coding: utf-8 -*-
"""
Módulo para administrar operaciones con archivos vectoriales (Shapefile, GeoPackage, KML).
"""

import os
import geopandas as gpd
import fiona
from typing import Optional, Dict

from core.logger import get_logger, log_execution_time
from core.exceptions import InvalidCRSError, UnsupportedFormatError
from .vector_metadata import VectorMetadata
from .vector_utils import parse_kml_fallback

logger = get_logger(__name__)

# Registrar soporte KML en fiona si es posible
try:
    if 'KML' not in fiona.drvsupport.supported_drivers:
        fiona.drvsupport.supported_drivers['KML'] = 'rw'
    if 'LIBKML' not in fiona.drvsupport.supported_drivers:
        fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
except Exception as e:
    logger.warning(f"No se pudieron registrar los drivers KML en Fiona: {e}")

class VectorManager:
    """Clase responsable de abrir, leer y analizar archivos vectoriales."""

    def __init__(self):
        self.file_path: Optional[str] = None
        self.gdf: Optional[gpd.GeoDataFrame] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @log_execution_time(logger)
    def open(self, file_path: str) -> None:
        """
        Abre un archivo vectorial utilizando geopandas.
        Soporta Shapefile (.shp), GeoPackage (.gpkg), GeoJSON (.geojson) y KML (.kml).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo vectorial no existe: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        supported = ['.shp', '.gpkg', '.geojson', '.kml']
        if ext not in supported:
            raise UnsupportedFormatError(
                f"Formato vectorial no soportado: '{ext}'. Formatos válidos: {supported}"
            )

        self.file_path = file_path
        
        try:
            if ext == '.kml':
                # Intentar leer con geopandas directamente
                try:
                    self.gdf = gpd.read_file(file_path)
                except Exception as kml_err:
                    logger.warning(f"Fallo al leer KML nativamente ({kml_err}). Usando parser alternativo...")
                    self.gdf = parse_kml_fallback(file_path)
            else:
                self.gdf = gpd.read_file(file_path)
                
            # Normalizar nombres de columnas (por ejemplo, 'Name' en KML) a 'name'
            if self.gdf is not None:
                if 'Name' in self.gdf.columns and 'name' not in self.gdf.columns:
                    self.gdf = self.gdf.rename(columns={'Name': 'name'})

            logger.info(f"Capa vectorial abierta correctamente: {file_path}")
        except Exception as e:
            logger.error(f"Error al abrir el archivo vectorial: {e}")
            raise

    def close(self) -> None:
        """Cierra/libera la capa vectorial de la memoria."""
        self.file_path = None
        self.gdf = None

    def get_features(self) -> gpd.GeoDataFrame:
        """Retorna el GeoDataFrame cargado."""
        if self.gdf is None:
            raise RuntimeError("Debe abrir un archivo vectorial primero.")
        return self.gdf

    def get_metadata(self) -> VectorMetadata:
        """
        Extrae y calcula los metadatos de la capa vectorial.
        Si la capa está en coordenadas geográficas, realiza una proyección
        al huso UTM local estimado para calcular el área y perímetro métrico real.
        """
        if self.gdf is None or not self.file_path:
            raise RuntimeError("Debe abrir un archivo vectorial primero.")

        crs_str = self.gdf.crs.to_string() if self.gdf.crs else "Sin CRS"
        feature_count = len(self.gdf)
        
        # Bounding box
        bounds_tuple = tuple(self.gdf.total_bounds)  # (minx, miny, maxx, maxy)

        # Campos y tipos de datos (excluyendo geometría)
        fields = {col: str(dtype) for col, dtype in self.gdf.dtypes.items() if col != 'geometry'}

        # Obtener tipo de geometría principal
        geom_types = self.gdf.geometry.type.unique()
        geom_type = geom_types[0] if len(geom_types) > 0 else "None"

        # Calcular área y perímetro proyectando a métrico si es geográfico
        if self.gdf.crs and self.gdf.crs.is_geographic:
            try:
                utm_crs = self.gdf.estimate_utm_crs()
                projected_gdf = self.gdf.to_crs(utm_crs)
            except Exception as e:
                logger.warning(f"No se pudo estimar el CRS UTM automáticamente ({e}). Usando EPSG:3857 de respaldo.")
                projected_gdf = self.gdf.to_crs("EPSG:3857")
        else:
            projected_gdf = self.gdf

        total_area = float(projected_gdf.geometry.area.sum())
        total_perimeter = float(projected_gdf.geometry.length.sum())

        return VectorMetadata(
            name=os.path.basename(self.file_path),
            path=os.path.abspath(self.file_path),
            geom_type=geom_type,
            feature_count=feature_count,
            crs=crs_str,
            fields=fields,
            bounds=bounds_tuple,
            total_area=total_area,
            total_perimeter=total_perimeter
        )
