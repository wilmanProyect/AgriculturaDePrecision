# -*- coding: utf-8 -*-
"""
Módulo de inferencia del Motor IA (detección de plantas con YOLO11).
"""

from .detection_result import Detection
from .detector import PlantDetector
from .row_segmenter import RowModelOptions, RowSegmenter

__all__ = [
    'Detection',
    'PlantDetector',
    'RowModelOptions',
    'RowSegmenter'
]
