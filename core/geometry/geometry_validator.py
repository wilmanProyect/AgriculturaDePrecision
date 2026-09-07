# -*- coding: utf-8 -*-
"""
Módulo para validación y reparación de geometrías espaciales.
"""

from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from core.logger import get_logger
from core.exceptions import InvalidGeometryError

logger = get_logger(__name__)

class GeometryValidator:
    """Clase responsable de verificar y corregir la validez topológica de geometrías."""

    @staticmethod
    def is_valid(geom: BaseGeometry) -> bool:
        """
        Verifica si la geometría es válida según los estándares OGC.
        """
        if geom is None:
            return False
        return geom.is_valid

    @staticmethod
    def repair(geom: BaseGeometry) -> BaseGeometry:
        """
        Intenta reparar una geometría inválida (por ejemplo, con auto-intersecciones).
        Si la geometría es válida, la devuelve intacta.
        """
        if geom is None:
            raise InvalidGeometryError("No se puede reparar una geometría nula.")

        if geom.is_valid:
            return geom

        logger.warning("Geometría inválida detectada. Intentando reparación...")
        try:
            repaired = make_valid(geom)
            logger.info("Geometría reparada con éxito usando make_valid.")
            return repaired
        except Exception as e:
            logger.warning(f"Fallo al usar make_valid ({e}). Intentando buffer(0)...")
            try:
                repaired = geom.buffer(0.0)
                if repaired.is_valid:
                    logger.info("Geometría reparada con éxito usando buffer(0.0).")
                    return repaired
            except Exception as buffer_err:
                logger.error(f"Fallo al reparar usando buffer(0.0): {buffer_err}")
                
            raise InvalidGeometryError("No se pudo reparar la geometría inválida automáticamente.")
