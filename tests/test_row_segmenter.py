"""Tests para RowSegmenter: longitud por esqueleto y ruteo sembrado/no sembrado."""
import types
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from ai.inference.row_segmenter import (
    RowModelOptions, RowSegmenter, _infer_gaps_between_rows, _keypoints_line, _skeleton_line
)
from core.analysis.row_detection import RowOptions
from core.exceptions import ModelNotFoundError


def _weights_file(tmp_path):
    weights = tmp_path / "row_lines_seg.pt"
    weights.write_bytes(b"fake-weights")
    return str(weights)


def test_row_model_options_raises_when_weights_missing():
    with pytest.raises(ModelNotFoundError):
        RowModelOptions(weights_path="no_existe.pt").validate()


def test_skeleton_line_straight_mask_measures_expected_length():
    mask = np.zeros((40, 220), dtype=bool)
    mask[20, 10:200] = True  # 190 px de largo
    transform = from_origin(500000, 8200000, 0.05, 0.05)

    result = _skeleton_line(mask, transform, pixel_size_m=0.05, min_length_m=1.0)

    assert result is not None
    line, length_m = result
    assert isinstance(line, LineString)
    assert 9.0 < length_m < 9.6  # ~189 px * 0.05 m


def test_skeleton_line_below_min_length_is_discarded():
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 5:8] = True  # muy corta
    transform = from_origin(500000, 8200000, 0.05, 0.05)

    assert _skeleton_line(mask, transform, pixel_size_m=0.05, min_length_m=5.0) is None


def _scene(tmp_path):
    height, width = 80, 160
    rgb = np.full((3, height, width), 100, dtype=np.uint8)
    path = tmp_path / "scene.tif"
    transform = from_origin(500000, 8200000, 0.05, 0.05)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=3,
                       dtype="uint8", crs="EPSG:32720", transform=transform, nodata=0) as dst:
        dst.write(rgb)
    polygon = box(500000.1, 8199993.9, 500007.9, 8199999.9)
    parcels = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs=32720)
    return str(path), parcels


def _fake_segment_masks(self, image):
    h, w = image.shape[:2]
    sown = np.zeros((h, w), dtype=bool)
    sown[h // 3, 3:w - 3] = True
    unsown = np.zeros((h, w), dtype=bool)
    unsown[2 * h // 3, 3:w - 3] = True
    return [
        (0, "linea_sembrada", 0.9, sown, None),
        (1, "linea_no_sembrada", 0.8, unsown, None),
    ]


def test_analyze_routes_instances_by_class_name(tmp_path):
    raster_path, parcels = _scene(tmp_path)

    with patch("ultralytics.YOLO", return_value=MagicMock(task="segment")):
        segmenter = RowSegmenter(RowModelOptions(weights_path=_weights_file(tmp_path)))
        segmenter.load_model()

    segmenter.segment = types.MethodType(_fake_segment_masks, segmenter)

    result = segmenter.analyze(raster_path, parcels, RowOptions(resolution_m=0.05, min_row_m=1.0))

    assert result["method"] == "YOLO11-segment"
    assert len(result["rows_gdf"]) == 1
    assert len(result["gaps_gdf"]) == 1
    assert result["rows_gdf"].iloc[0]["long_m"] > 1.0
    assert result["gaps_gdf"].iloc[0]["long_m"] > 1.0
    assert "linea_sembrada" in result["rows_gdf"].iloc[0]["estado"]
    assert "linea_no_sembrada" in result["gaps_gdf"].iloc[0]["estado"]


def test_analyze_handles_large_parcel_via_tiling_without_raising(tmp_path):
    """max_pixels chico para esta parcela dividía en piezas o tiraba 'demasiado
    grande' antes de tener tiling; ahora se resuelve dividiendo en piezas."""
    raster_path, parcels = _scene(tmp_path)

    with patch("ultralytics.YOLO", return_value=MagicMock(task="segment")):
        segmenter = RowSegmenter(RowModelOptions(weights_path=_weights_file(tmp_path)))
        segmenter.load_model()
    segmenter.segment = types.MethodType(_fake_segment_masks, segmenter)

    result = segmenter.analyze(raster_path, parcels,
                                RowOptions(resolution_m=0.05, min_row_m=1.0, max_pixels=5000))

    assert not result["rows_gdf"].empty
    assert any('.' in pid for pid in result["rows_gdf"]["parcela_id"])


def test_analyze_accepts_precomputed_crs_and_skips_estimate_utm_crs(tmp_path, monkeypatch):
    raster_path, parcels = _scene(tmp_path)
    crs = parcels.estimate_utm_crs()
    metric = parcels.to_crs(crs)

    def _boom(*args, **kwargs):
        raise AssertionError("no debería llamarse estimate_utm_crs cuando se pasa crs=")
    monkeypatch.setattr(gpd.GeoDataFrame, "estimate_utm_crs", _boom)

    with patch("ultralytics.YOLO", return_value=MagicMock(task="segment")):
        segmenter = RowSegmenter(RowModelOptions(weights_path=_weights_file(tmp_path)))
        segmenter.load_model()
    segmenter.segment = types.MethodType(_fake_segment_masks, segmenter)

    result = segmenter.analyze(raster_path, metric, RowOptions(resolution_m=0.05, min_row_m=1.0), crs=crs)
    assert not result["rows_gdf"].empty


def test_keypoints_line_measures_expected_length():
    valid = np.ones((40, 220), dtype=bool)
    transform = from_origin(500000, 8200000, 0.05, 0.05)
    kpts = np.array([[10.0, 20.0], [200.0, 20.0]])  # 190 px de separación en x

    result = _keypoints_line(kpts, valid, transform, min_length_m=1.0)

    assert result is not None
    line, length_m = result
    assert isinstance(line, LineString)
    assert 9.4 < length_m < 9.6  # 190 px * 0.05 m


def test_keypoints_line_discarded_outside_valid_area():
    valid = np.zeros((40, 220), dtype=bool)  # nada válido (fuera de la parcela / NoData)
    transform = from_origin(500000, 8200000, 0.05, 0.05)
    kpts = np.array([[10.0, 20.0], [200.0, 20.0]])

    assert _keypoints_line(kpts, valid, transform, min_length_m=1.0) is None


def _fake_segment_pose(self, image):
    h, w = image.shape[:2]
    return [
        (0, "Crop_Row", 0.9, None, np.array([[3.0, h / 3], [w - 3.0, h / 3]])),
        (0, "Crop_Row", 0.8, None, np.array([[3.0, 2 * h / 3], [w / 2 - 5, 2 * h / 3]])),
        (0, "Crop_Row", 0.8, None, np.array([[w / 2 + 5, 2 * h / 3], [w - 3.0, 2 * h / 3]])),
    ]


def test_analyze_with_pose_model_infers_gap_from_single_class(tmp_path):
    raster_path, parcels = _scene(tmp_path)

    with patch("ultralytics.YOLO", return_value=MagicMock(task="pose")):
        segmenter = RowSegmenter(RowModelOptions(weights_path=_weights_file(tmp_path)))
        segmenter.load_model()

    segmenter.segment = types.MethodType(_fake_segment_pose, segmenter)

    result = segmenter.analyze(raster_path, parcels, RowOptions(resolution_m=0.05, min_row_m=0.2, spacing_m=1.0))

    assert result["method"] == "YOLO11-pose"
    # 1 línea larga (fila de arriba) + 2 tramos cortos (fila de abajo, con un hueco en medio)
    assert len(result["rows_gdf"]) == 3
    # el hueco entre los dos tramos cortos de la fila de abajo se infiere sin clase de falla
    assert len(result["gaps_gdf"]) == 1
    assert "hueco entre detecciones" in result["gaps_gdf"].iloc[0]["estado"]


def test_infer_gaps_between_rows_finds_collinear_gap():
    same_row = [
        {'geometry': LineString([(0, 0), (10, 0)])},
        {'geometry': LineString([(15, 0.1), (25, 0.1)])},  # hueco de 5 m, misma fila
    ]
    other_row = [{'geometry': LineString([(0, 5), (10, 5)])}]  # fila distinta, sin pareja

    gaps = _infer_gaps_between_rows(same_row + other_row, min_gap_m=1.0, max_lateral_offset_m=0.5)

    assert len(gaps) == 1
    assert 4.5 < gaps[0]['long_m'] < 5.5


def test_infer_gaps_between_rows_ignores_small_gaps():
    close_together = [
        {'geometry': LineString([(0, 0), (10, 0)])},
        {'geometry': LineString([(10.3, 0), (20, 0)])},
    ]
    assert _infer_gaps_between_rows(close_together, min_gap_m=1.0, max_lateral_offset_m=0.5) == []


def test_analyze_raises_without_loaded_model(tmp_path):
    raster_path, parcels = _scene(tmp_path)
    segmenter = RowSegmenter.__new__(RowSegmenter)  # skip __init__/validate: solo probamos el guard de carga
    segmenter.options = RowModelOptions(weights_path=_weights_file(tmp_path))
    segmenter._model = None
    with pytest.raises(RuntimeError):
        segmenter.analyze(raster_path, parcels)
