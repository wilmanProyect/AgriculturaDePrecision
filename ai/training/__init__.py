# -*- coding: utf-8 -*-
"""
Módulo de entrenamiento del Motor IA (YOLO11).
"""

from .dataset_config import DatasetConfig
from .trainer import TrainingConfig, YOLOTrainer

__all__ = [
    'DatasetConfig',
    'TrainingConfig',
    'YOLOTrainer'
]
