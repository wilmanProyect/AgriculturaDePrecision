# -*- coding: utf-8 -*-
"""
Pruebas unitarias para PlantDetector (inferencia YOLO11).
El modelo de Ultralytics se sustituye por dobles de prueba: estas pruebas
verifican la lógica propia del detector (tiling, offsets, georreferenciación,
parseo de resultados), no el comportamiento interno de YOLO11.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ai.inference.detection_result import Detection
from ai.inference.detector import PlantDetector
from core.exceptions import ModelNotFoundError
from core.gis.raster.raster_manager import RasterManager


class _FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.array([xyxy], dtype=np.float64)
        self.conf = np.array([conf], dtype=np.float64)
        self.cls = np.array([cls], dtype=np.float64)


class _FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


@pytest.fixture
def tiny_raster(tmp_path):
    """Genera un GeoTIFF de 20x20 píxeles de una sola banda para probar el tiling."""
    raster_path = tmp_path / "mini_ortomosaico.tif"
    transform = from_origin(-4.0, 40.5, 0.01, 0.01)
    with rasterio.open(
        str(raster_path), "w", driver="GTiff",
        height=20, width=20, count=1, dtype="uint8",
        crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(np.ones((20, 20), dtype="uint8") * 50, 1)
    return str(raster_path)


def test_load_model_missing_weights_raises(tmp_path):
    detector = PlantDetector(weights_path=str(tmp_path / "no_existe.pt"))
    with pytest.raises(ModelNotFoundError):
        detector.load_model()
    assert detector.is_loaded is False


def test_load_model_success(tmp_path):
    weights_path = tmp_path / "fake_weights.pt"
    weights_path.write_bytes(b"fake")

    detector = PlantDetector(weights_path=str(weights_path), device="cpu")
    with patch("ultralytics.YOLO", return_value=MagicMock()) as mock_yolo:
        detector.load_model()

    mock_yolo.assert_called_once_with(str(weights_path))
    assert detector.is_loaded is True


def test_predict_without_loaded_model_raises():
    detector = PlantDetector(weights_path="irrelevante.pt")
    with pytest.raises(RuntimeError):
        detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))


def test_predict_parses_fake_results_into_detections():
    detector = PlantDetector(weights_path="irrelevante.pt")
    fake_box = _FakeBox(xyxy=(2.0, 4.0, 8.0, 10.0), conf=0.87, cls=0)
    fake_result = _FakeResult(boxes=[fake_box], names={0: "planta"})

    detector._model = MagicMock()
    detector._model.predict.return_value = [fake_result]

    detections = detector.predict(np.zeros((20, 20, 3), dtype=np.uint8))

    assert len(detections) == 1
    d = detections[0]
    assert d.class_id == 0
    assert d.class_name == "planta"
    assert d.confidence == pytest.approx(0.87)
    assert d.bbox_pixel == (2.0, 4.0, 8.0, 10.0)
    assert d.center_pixel == (5.0, 7.0)
    assert d.center_geo is None


def test_predict_orthomosaic_offsets_tiles_and_georeferences(tiny_raster, monkeypatch):
    detector = PlantDetector(weights_path="irrelevante.pt")
    detector._model = MagicMock()  # simula modelo cargado

    call_count = {"n": 0}

    def fake_predict(image):
        call_count["n"] += 1
        return [Detection(
            class_id=0, class_name="planta", confidence=0.9,
            bbox_pixel=(1.0, 1.0, 3.0, 3.0), center_pixel=(2.0, 2.0)
        )]

    monkeypatch.setattr(detector, "predict", fake_predict)

    with RasterManager() as raster_manager:
        raster_manager.open(tiny_raster)
        # Raster de 20x20, tiles de 10x10 sin solape -> grilla 2x2 = 4 tiles
        detections = detector.predict_orthomosaic(raster_manager, tile_size=10, overlap=0.0)

    assert call_count["n"] == 4
    # Cada detección local se desplaza a una posición global distinta -> ninguna se fusiona
    assert len(detections) == 4
    assert all(d.center_geo is not None for d in detections)

    global_centers = sorted(d.center_pixel for d in detections)
    assert global_centers == [(2.0, 2.0), (2.0, 12.0), (12.0, 2.0), (12.0, 12.0)]


def test_predict_orthomosaic_reports_progress(tiny_raster, monkeypatch):
    detector = PlantDetector(weights_path="irrelevante.pt")
    detector._model = MagicMock()
    monkeypatch.setattr(detector, "predict", lambda image: [])

    progress_calls = []

    with RasterManager() as raster_manager:
        raster_manager.open(tiny_raster)
        detector.predict_orthomosaic(
            raster_manager, tile_size=10, overlap=0.0,
            progress_callback=lambda done, total: progress_calls.append((done, total))
        )

    assert progress_calls == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_predict_orthomosaic_bounds_restricts_to_roi(tiny_raster, monkeypatch):
    """Con `bounds`, solo se deben tilear los pixeles dentro de esa región, no todo el ráster."""
    detector = PlantDetector(weights_path="irrelevante.pt")
    detector._model = MagicMock()
    monkeypatch.setattr(detector, "predict", lambda image: [])

    call_count = {"n": 0}
    original_predict = detector.predict

    def counting_predict(image):
        call_count["n"] += 1
        return original_predict(image)

    monkeypatch.setattr(detector, "predict", counting_predict)

    with RasterManager() as raster_manager:
        raster_manager.open(tiny_raster)
        # Raster completo de 20x20 -> normalmente 4 tiles de 10x10.
        # bounds cubre solo el cuadrante superior izquierdo -> debe procesar solo 1 tile.
        detector.predict_orthomosaic(
            raster_manager, tile_size=10, overlap=0.0,
            bounds=(-4.0, 40.4, -3.9, 40.5)
        )

    assert call_count["n"] == 1


def test_predict_orthomosaic_should_stop_cancels_early(tiny_raster, monkeypatch):
    detector = PlantDetector(weights_path="irrelevante.pt")
    detector._model = MagicMock()

    call_count = {"n": 0}

    def fake_predict(image):
        call_count["n"] += 1
        return []

    monkeypatch.setattr(detector, "predict", fake_predict)

    with RasterManager() as raster_manager:
        raster_manager.open(tiny_raster)
        # 4 tiles en total; cancelar tras el primero procesado.
        detector.predict_orthomosaic(
            raster_manager, tile_size=10, overlap=0.0,
            should_stop=lambda: call_count["n"] >= 1
        )

    assert call_count["n"] == 1


def test_predict_orthomosaic_requires_open_raster():
    detector = PlantDetector(weights_path="irrelevante.pt")
    detector._model = MagicMock()
    empty_manager = RasterManager()

    with pytest.raises(RuntimeError):
        detector.predict_orthomosaic(empty_manager)


def test_count_by_class():
    detections = [
        Detection(0, "planta", 0.9, (0, 0, 1, 1), (0.5, 0.5)),
        Detection(0, "planta", 0.8, (0, 0, 1, 1), (0.5, 0.5)),
        Detection(1, "arbol", 0.7, (0, 0, 1, 1), (0.5, 0.5)),
    ]
    counts = PlantDetector.count_by_class(detections)
    assert counts == {"planta": 2, "arbol": 1}
