# -*- coding: utf-8 -*-
"""
Módulo para realizar cálculos geométricos métricos sobre objetos espaciales.
"""

from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
import pyproj
from core.logger import get_logger
from core.exceptions import InvalidCRSError

logger = get_logger(__name__)

class GeometryManager:
    """Clase responsable de realizar mediciones espaciales y transformaciones de geometrías."""

    @staticmethod
    def get_utm_crs_for_point(lon: float, lat: float) -> str:
        """
        Determina la zona UTM correcta (WGS 84 / UTM zone) basándose en la longitud y latitud.
        Retorna el código EPSG correspondiente.
        """
        # Limitar longitud en rango [-180, 180]
        lon = max(min(lon, 180.0), -180.0)
        lat = max(min(lat, 90.0), -90.0)
        
        zone = int((lon + 180) / 6) + 1
        if zone > 60:
            zone = 60

        if lat >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone
        return f"EPSG:{epsg}"

    @staticmethod
    def project_geometry(geom: BaseGeometry, from_crs: str, to_crs: str) -> BaseGeometry:
        """
        Reproyecta una geometría individual de un CRS a otro.
        """
        try:
            transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
            reprojected = transform(transformer.transform, geom)
            return reprojected
        except Exception as e:
            raise InvalidCRSError(f"No se pudo reproyectar la geometría de '{from_crs}' a '{to_crs}': {e}")

    @classmethod
    def calculate_area(cls, geom: BaseGeometry, crs: str) -> float:
        """
        Calcula el área en metros cuadrados. Si el CRS es geográfico (grados),
        proyecta la geometría al huso UTM estimado para obtener una medición métrica real.
        """
        if geom is None:
            return 0.0

        pyproj_crs = pyproj.CRS(crs)
        if pyproj_crs.is_geographic:
            centroid = geom.centroid
            utm_crs = cls.get_utm_crs_for_point(centroid.x, centroid.y)
            logger.debug(f"Reproyectando geometría para cálculo de área de {crs} a {utm_crs}")
            projected_geom = cls.project_geometry(geom, crs, utm_crs)
            return float(projected_geom.area)
        
        return float(geom.area)

    @classmethod
    def calculate_perimeter(cls, geom: BaseGeometry, crs: str) -> float:
        """
        Calcula el perímetro en metros. Si el CRS es geográfico,
        proyecta la geometría al huso UTM estimado para obtener una medición métrica real.
        """
        if geom is None:
            return 0.0

        pyproj_crs = pyproj.CRS(crs)
        if pyproj_crs.is_geographic:
            centroid = geom.centroid
            utm_crs = cls.get_utm_crs_for_point(centroid.x, centroid.y)
            logger.debug(f"Reproyectando geometría para cálculo de perímetro de {crs} a {utm_crs}")
            projected_geom = cls.project_geometry(geom, crs, utm_crs)
            return float(projected_geom.length)
        
        return float(geom.length)

    @staticmethod
    def calculate_centroid(geom: BaseGeometry) -> tuple:
        """
        Calcula y retorna el centroide de la geometría como una tupla (x, y).
        """
        if geom is None:
            raise ValueError("No se puede obtener el centroide de una geometría nula.")
        centroid = geom.centroid
        return (centroid.x, centroid.y)
