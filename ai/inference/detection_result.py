# -*- coding: utf-8 -*-
"""
Estructuras de datos para representar detecciones de YOLO11.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional


@dataclass
class Detection:
    """Representa una única detección (planta, árbol o espacio vacío) de YOLO11."""

    class_id: int                              # Índice de clase devuelto por el modelo
    class_name: str                            # Nombre de la clase (ej. 'planta')
    confidence: float                          # Confianza de la predicción (0-1)
    bbox_pixel: Tuple[float, float, float, float]  # Bounding box (x1, y1, x2, y2) en píxeles del ortomosaico
    center_pixel: Tuple[float, float]          # Centro (col, row) en píxeles del ortomosaico
    center_geo: Optional[Tuple[float, float]] = None  # Centro (x, y) georreferenciado en el CRS del ráster

    def to_dict(self) -> dict:
        """Convierte la detección a un diccionario estándar."""
        return asdict(self)
