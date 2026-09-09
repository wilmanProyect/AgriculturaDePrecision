# -*- coding: utf-8 -*-
"""
Pruebas unitarias para YOLOTrainer (entrenamiento de YOLO11).
Ultralytics se sustituye por un doble de prueba: verificamos que YOLOTrainer
propague correctamente la configuración y maneje los errores esperados.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from ai.training.trainer import TrainingConfig, YOLOTrainer
from core.exceptions import InvalidDatasetError, TrainingError


def _config(data_yaml: str) -> TrainingConfig:
    return TrainingConfig(
        data_yaml=data_yaml,
        model_arch="yolo11n.pt",
        epochs=5,
        imgsz=320,
        batch=4,
        device="cpu",
        project="data/models",
        name="test_run"
    )


def test_train_raises_when_data_yaml_missing(tmp_path):
    trainer = YOLOTrainer(_config(str(tmp_path / "no_existe.yaml")))
    with pytest.raises(InvalidDatasetError):
        trainer.train()


def test_train_calls_ultralytics_with_expected_params(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\n", encoding="utf-8")

    trainer = YOLOTrainer(_config(str(data_yaml)))

    mock_model_instance = MagicMock()
    with patch("ultralytics.YOLO", return_value=mock_model_instance) as mock_yolo:
        trainer.train()

    mock_yolo.assert_called_once_with("yolo11n.pt")
    mock_model_instance.train.assert_called_once_with(
        data=str(data_yaml), epochs=5, imgsz=320, batch=4,
        device="cpu", patience=50, project=os.path.abspath("data/models"), name="test_run",
        workers=4
    )


def test_train_wraps_ultralytics_failures_in_training_error(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\n", encoding="utf-8")

    trainer = YOLOTrainer(_config(str(data_yaml)))

    mock_model_instance = MagicMock()
    mock_model_instance.train.side_effect = RuntimeError("fallo simulado de entrenamiento")

    with patch("ultralytics.YOLO", return_value=mock_model_instance):
        with pytest.raises(TrainingError):
            trainer.train()


def test_validate_requires_trained_model(tmp_path):
    trainer = YOLOTrainer(_config(str(tmp_path / "data.yaml")))
    with pytest.raises(RuntimeError):
        trainer.validate()


def test_export_requires_trained_model(tmp_path):
    trainer = YOLOTrainer(_config(str(tmp_path / "data.yaml")))
    with pytest.raises(RuntimeError):
        trainer.export()
