# -*- coding: utf-8 -*-
"""
Módulo de inferencia del Motor IA (detección de plantas con YOLO11).
"""

from .detection_result import Detection
from .detector import PlantDetector

__all__ = [
    'Detection',
    'PlantDetector'
]
