# -*- coding: utf-8 -*-
"""
Estructura de metadatos para archivos ráster.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional

@dataclass
class RasterMetadata:
    """Clase de datos para almacenar los metadatos de un archivo ráster."""
    name: str                  # Nombre del archivo
    path: str                  # Ruta completa
    width: int                 # Ancho en píxeles
    height: int                # Alto en píxeles
    count: int                 # Número de bandas
    crs: str                   # Sistema de Referencia de Coordenadas (EPSG o WKT)
    res_x: float               # Resolución espacial en X (tamaño de píxel)
    res_y: float               # Resolución espacial en Y (tamaño de píxel)
    transform: Tuple[float, ...]  # Matriz de transformación afín (valores de Affine)
    bounds: Tuple[float, float, float, float]  # Bounding Box (min_x, min_y, max_x, max_y)
    dtype: str                 # Tipo de dato de los píxeles (uint8, float32, etc.)
    nodata: Optional[float]    # Valor definido para representar NoData
    size_bytes: int            # Tamaño del archivo en disco en bytes

    def to_dict(self) -> dict:
        """Convierte los metadatos a un diccionario estándar."""
        return asdict(self)
