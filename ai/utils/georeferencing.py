# -*- coding: utf-8 -*-
"""
Módulo para convertir detecciones en píxeles a coordenadas geográficas
y empaquetarlas como capas vectoriales listas para el Motor GIS.

La conversión píxel -> coordenadas reutiliza `core.gis.raster.pixel_to_coords`
para no duplicar lógica de georreferenciación entre el Motor GIS y el Motor IA.
"""

from typing import List, Tuple

import geopandas as gpd
from shapely.geometry import Point

from core.gis.raster import pixel_to_coords
from core.logger import get_logger

logger = get_logger(__name__)


def georeference_detections(detections: List["Detection"], transform: Tuple[float, ...]) -> List["Detection"]:
    """
    Rellena el campo `center_geo` de cada detección a partir de su centro en píxeles
    (col, row) y la transformación afín del ráster de origen (tupla de 6-9 valores,
    tal como la expone `rasterio.DatasetReader.transform`). Devuelve la misma lista, modificada in-place.
    """
    for detection in detections:
        col, row = detection.center_pixel
        x, y = pixel_to_coords(row, col, transform)
        detection.center_geo = (x, y)
    return detections


def detections_to_geodataframe(detections: List["Detection"], crs: str) -> gpd.GeoDataFrame:
    """
    Construye una capa de puntos (GeoDataFrame) a partir de una lista de detecciones
    georreferenciadas, lista para cruzarse con parcelas mediante SpatialAnalysis.point_in_polygon.
    """
    if not detections:
        logger.info("No hay detecciones para convertir a GeoDataFrame.")
        return gpd.GeoDataFrame(
            {"class_id": [], "class_name": [], "confidence": []},
            geometry=[],
            crs=crs
        )

    records = []
    geometries = []
    for detection in detections:
        if detection.center_geo is None:
            raise ValueError(
                "La detección no está georreferenciada. "
                "Ejecute georeference_detections() antes de construir el GeoDataFrame."
            )
        x, y = detection.center_geo
        geometries.append(Point(x, y))
        records.append({
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence
        })

    return gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
