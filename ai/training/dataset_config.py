# -*- coding: utf-8 -*-
"""
Construcción y validación de la estructura de dataset YOLO
(images/labels, train/val) y su archivo data.yaml.
"""

import os
from typing import List, Optional

import yaml

from core.exceptions import InvalidDatasetError
from core.logger import get_logger

logger = get_logger(__name__)

_SUBDIRS = (
    os.path.join("images", "train"),
    os.path.join("images", "val"),
    os.path.join("labels", "train"),
    os.path.join("labels", "val"),
)


class DatasetConfig:
    """Clase responsable de generar y validar la estructura de un dataset de entrenamiento YOLO11."""

    def __init__(self, dataset_dir: str, class_names: List[str]):
        if not class_names:
            raise InvalidDatasetError("Debe especificar al menos una clase (ej. ['planta', 'arbol']).")

        self.dataset_dir = dataset_dir
        self.class_names = class_names

    def build_directory_structure(self) -> None:
        """Crea la estructura de carpetas images/{train,val} y labels/{train,val} si no existe."""
        for subdir in _SUBDIRS:
            os.makedirs(os.path.join(self.dataset_dir, subdir), exist_ok=True)
        logger.info(f"Estructura de dataset preparada en: {self.dataset_dir}")

    def write_data_yaml(self, output_path: Optional[str] = None) -> str:
        """
        Escribe el archivo data.yaml que Ultralytics necesita para entrenar,
        con las rutas de train/val y el mapeo de nombres de clase.
        """
        output_path = output_path or os.path.join(self.dataset_dir, "data.yaml")

        data = {
            "path": os.path.abspath(self.dataset_dir),
            "train": os.path.join("images", "train"),
            "val": os.path.join("images", "val"),
            "names": {idx: name for idx, name in enumerate(self.class_names)}
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

        logger.info(f"Archivo data.yaml generado en: {output_path}")
        return output_path

    def validate(self) -> None:
        """
        Verifica que el dataset tenga la estructura mínima esperada y contenga
        al menos una imagen de entrenamiento. Lanza InvalidDatasetError en caso contrario.
        """
        for subdir in _SUBDIRS:
            full_path = os.path.join(self.dataset_dir, subdir)
            if not os.path.isdir(full_path):
                raise InvalidDatasetError(f"Falta la carpeta requerida del dataset: {full_path}")

        train_images_dir = os.path.join(self.dataset_dir, "images", "train")
        has_images = any(
            entry.is_file() for entry in os.scandir(train_images_dir)
        )
        if not has_images:
            raise InvalidDatasetError(
                f"No se encontraron imágenes de entrenamiento en: {train_images_dir}"
            )
