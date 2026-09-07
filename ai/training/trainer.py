# -*- coding: utf-8 -*-
"""
Entrenamiento de modelos YOLO11 (Ultralytics) para detección de plantas,
árboles y espacios vacíos sobre ortomosaicos.
"""

import os
from dataclasses import dataclass

from core.exceptions import InvalidDatasetError, TrainingError
from core.logger import get_logger, log_execution_time

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuración de un entrenamiento YOLO11."""

    data_yaml: str                     # Ruta al data.yaml del dataset (ver DatasetConfig)
    model_arch: str = "yolo11n.pt"     # Arquitectura/checkpoint base (yolo11n/s/m/l/x.pt)
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"
    patience: int = 50                 # Épocas sin mejora antes de early stopping
    project: str = "data/models"       # Carpeta donde Ultralytics guarda las corridas
    name: str = "plant_detector"       # Nombre de la corrida (project/name/weights/best.pt)


class YOLOTrainer:
    """Clase responsable de entrenar, validar y exportar un modelo YOLO11."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._model = None

    @log_execution_time(logger)
    def train(self):
        """
        Entrena un modelo YOLO11 usando el dataset y los hiperparámetros de `config`.
        Devuelve el objeto de resultados de Ultralytics con las métricas de entrenamiento.
        """
        if not os.path.exists(self.config.data_yaml):
            raise InvalidDatasetError(f"No se encontró el archivo data.yaml: {self.config.data_yaml}")

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.config.model_arch)
            results = self._model.train(
                data=self.config.data_yaml,
                epochs=self.config.epochs,
                imgsz=self.config.imgsz,
                batch=self.config.batch,
                device=self.config.device,
                patience=self.config.patience,
                project=self.config.project,
                name=self.config.name
            )
            logger.info(f"Entrenamiento finalizado. Resultados en: {self.config.project}/{self.config.name}")
            return results
        except InvalidDatasetError:
            raise
        except Exception as e:
            raise TrainingError(f"Fallo durante el entrenamiento de YOLO11: {e}") from e

    @log_execution_time(logger)
    def validate(self):
        """Valida el modelo recién entrenado (o cargado) sobre el split de validación del dataset."""
        if self._model is None:
            raise RuntimeError("Debe entrenar o cargar un modelo antes de validarlo.")
        try:
            return self._model.val(data=self.config.data_yaml, device=self.config.device)
        except Exception as e:
            raise TrainingError(f"Fallo durante la validación del modelo: {e}") from e

    def export(self, format: str = "onnx") -> str:
        """Exporta el modelo entrenado a otro formato (ej. 'onnx') para despliegue."""
        if self._model is None:
            raise RuntimeError("Debe entrenar o cargar un modelo antes de exportarlo.")
        try:
            return self._model.export(format=format)
        except Exception as e:
            raise TrainingError(f"Fallo al exportar el modelo a '{format}': {e}") from e
