# -*- coding: utf-8 -*-
"""
Módulo para convertir detecciones en píxeles a coordenadas geográficas
y empaquetarlas como capas vectoriales listas para el Motor GIS.

La conversión píxel -> coordenadas reutiliza `core.gis.raster.pixel_to_coords`
para no duplicar lógica de georreferenciación entre el Motor GIS y el Motor IA.
"""

from typing import List, Tuple

import geopandas as gpd
from shapely.geometry import Point, Polygon

from core.gis.raster import pixel_to_coords
from core.logger import get_logger

logger = get_logger(__name__)


def georeference_detections(detections: List["Detection"], transform: Tuple[float, ...]) -> List["Detection"]:
    """
    Rellena `center_geo` y `bbox_geo` de cada detección a partir de su centro y su bounding
    box en píxeles, y la transformación afín del ráster de origen (tupla de 6-9 valores,
    tal como la expone `rasterio.DatasetReader.transform`). Devuelve la misma lista, modificada in-place.
    `bbox_geo` guarda las 4 esquinas (x, y) del recuadro, en el mismo orden que las esquinas en
    píxeles (no solo min/max), para poder reconstruir el polígono correctamente aunque el
    ráster tenga rotación en su transformación afín.
    """
    for detection in detections:
        col, row = detection.center_pixel
        x, y = pixel_to_coords(row, col, transform)
        detection.center_geo = (x, y)

        x1, y1, x2, y2 = detection.bbox_pixel
        corners_pixel = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        detection.bbox_geo = tuple(pixel_to_coords(py, px, transform) for px, py in corners_pixel)
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


def detections_to_polygon_geodataframe(detections: List["Detection"], crs: str) -> gpd.GeoDataFrame:
    """
    Construye una capa de polígonos (GeoDataFrame) a partir de una lista de detecciones
    georreferenciadas, dibujando el bounding box de cada detección (el recuadro que YOLO
    ajustó alrededor del objeto) en lugar de solo su punto central. Útil para visualizar
    el contorno detectado (ej. malezas) en vez de un único punto.
    """
    if not detections:
        logger.info("No hay detecciones para convertir a GeoDataFrame de polígonos.")
        return gpd.GeoDataFrame(
            {"class_id": [], "class_name": [], "confidence": []},
            geometry=[],
            crs=crs
        )

    records = []
    geometries = []
    for detection in detections:
        if detection.bbox_geo is None:
            raise ValueError(
                "La detección no está georreferenciada. "
                "Ejecute georeference_detections() antes de construir el GeoDataFrame."
            )
        geometries.append(Polygon(detection.bbox_geo))
        records.append({
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence
        })

    return gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
