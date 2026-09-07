# -*- coding: utf-8 -*-
"""
Estructura de metadatos para archivos vectoriales.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Dict

@dataclass
class VectorMetadata:
    """Clase de datos para almacenar los metadatos de una capa vectorial."""
    name: str                  # Nombre del archivo/capa
    path: str                  # Ruta completa al archivo
    geom_type: str             # Tipo de geometría principal (Polygon, Point, etc.)
    feature_count: int         # Número de entidades (features)
    crs: str                   # Sistema de Referencia de Coordenadas (EPSG o WKT)
    fields: Dict[str, str]     # Nombres y tipos de atributos (esquema)
    bounds: Tuple[float, float, float, float]  # Bounding Box (min_x, min_y, max_x, max_y)
    total_area: float          # Área total de las geometrías en metros cuadrados
    total_perimeter: float     # Perímetro total en metros

    def to_dict(self) -> dict:
        """Convierte los metadatos a un diccionario estándar."""
        return asdict(self)
