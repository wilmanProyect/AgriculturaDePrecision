# -*- coding: utf-8 -*-
"""
Funciones auxiliares para manipulación y transformación de datos ráster.
"""

from typing import Tuple
import rasterio.transform

def pixel_to_coords(row: int, col: int, transform: Tuple[float, ...]) -> Tuple[float, float]:
    """
    Convierte coordenadas de píxel (fila, columna) a coordenadas geográficas/proyectadas (x, y)
    utilizando la matriz de transformación afín del ráster.
    """
    # transform debe ser convertida a Affine si no lo es
    affine_transform = rasterio.transform.Affine(*transform[:6])
    x, y = rasterio.transform.xy(affine_transform, row, col)
    return x, y

def coords_to_pixel(x: float, y: float, transform: Tuple[float, ...]) -> Tuple[int, int]:
    """
    Convierte coordenadas geográficas/proyectadas (x, y) a coordenadas de píxel (fila, columna).
    """
    import math
    affine_transform = rasterio.transform.Affine(*transform[:6])
    inv_transform = ~affine_transform
    col_val, row_val = inv_transform * (x, y)
    return int(math.floor(row_val)), int(math.floor(col_val))

