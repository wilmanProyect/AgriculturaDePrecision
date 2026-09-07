# -*- coding: utf-8 -*-
"""
Pruebas unitarias para la construcción y validación de datasets de entrenamiento YOLO11.
"""

import os

import pytest
import yaml

from ai.training.dataset_config import DatasetConfig
from core.exceptions import InvalidDatasetError


def test_dataset_config_requires_class_names():
    with pytest.raises(InvalidDatasetError):
        DatasetConfig("some/dir", [])


def test_build_directory_structure_creates_expected_folders(tmp_path):
    dataset_dir = str(tmp_path / "dataset")
    config = DatasetConfig(dataset_dir, ["planta", "arbol", "espacio_vacio"])

    config.build_directory_structure()

    assert os.path.isdir(os.path.join(dataset_dir, "images", "train"))
    assert os.path.isdir(os.path.join(dataset_dir, "images", "val"))
    assert os.path.isdir(os.path.join(dataset_dir, "labels", "train"))
    assert os.path.isdir(os.path.join(dataset_dir, "labels", "val"))


def test_write_data_yaml_contains_expected_fields(tmp_path):
    dataset_dir = str(tmp_path / "dataset")
    config = DatasetConfig(dataset_dir, ["planta", "arbol"])
    config.build_directory_structure()

    yaml_path = config.write_data_yaml()

    assert os.path.exists(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["names"] == {0: "planta", 1: "arbol"}
    assert data["train"] == os.path.join("images", "train")
    assert data["val"] == os.path.join("images", "val")


def test_validate_raises_when_structure_missing(tmp_path):
    dataset_dir = str(tmp_path / "dataset")
    config = DatasetConfig(dataset_dir, ["planta"])

    with pytest.raises(InvalidDatasetError):
        config.validate()


def test_validate_raises_when_no_training_images(tmp_path):
    dataset_dir = str(tmp_path / "dataset")
    config = DatasetConfig(dataset_dir, ["planta"])
    config.build_directory_structure()

    with pytest.raises(InvalidDatasetError):
        config.validate()


def test_validate_passes_with_at_least_one_image(tmp_path):
    dataset_dir = str(tmp_path / "dataset")
    config = DatasetConfig(dataset_dir, ["planta"])
    config.build_directory_structure()

    open(os.path.join(dataset_dir, "images", "train", "img_0001.jpg"), "wb").close()

    config.validate()  # No debe lanzar excepción
