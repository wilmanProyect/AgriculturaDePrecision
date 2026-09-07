# -*- coding: utf-8 -*-
"""
Pruebas unitarias para la georreferenciación de detecciones del Motor IA.
"""

import pytest
from affine import Affine

from ai.inference.detection_result import Detection
from ai.utils.georeferencing import detections_to_geodataframe, georeference_detections


def _transform():
    # Origen en (-4.0, 40.5), tamaño de píxel de 0.01 grados (igual que tests/generate_test_data.py)
    from rasterio.transform import from_origin
    return tuple(from_origin(-4.0, 40.5, 0.01, 0.01))


def _make_detection(center_pixel):
    return Detection(
        class_id=0,
        class_name="planta",
        confidence=0.9,
        bbox_pixel=(0, 0, 10, 10),
        center_pixel=center_pixel
    )


def test_georeference_detections_fills_center_geo():
    transform = _transform()
    detection = _make_detection((0, 0))

    result = georeference_detections([detection], transform)

    # rasterio.transform.xy() devuelve el CENTRO del píxel (0,0), no su esquina
    assert result[0].center_geo == pytest.approx((-3.995, 40.495))


def test_georeference_detections_offsets_with_pixel_position():
    transform = _transform()
    # A 10 píxeles a la derecha y 10 hacia abajo del origen
    detection = _make_detection((10, 10))

    result = georeference_detections([detection], transform)

    assert result[0].center_geo == pytest.approx((-3.895, 40.395))


def test_detections_to_geodataframe_builds_points():
    transform = _transform()
    detections = georeference_detections([_make_detection((0, 0)), _make_detection((10, 10))], transform)

    gdf = detections_to_geodataframe(detections, crs="EPSG:4326")

    assert len(gdf) == 2
    assert gdf.crs.to_string() == "EPSG:4326"
    assert list(gdf.columns).count("geometry") == 1
    assert gdf.iloc[0].geometry.x == pytest.approx(-3.995)


def test_detections_to_geodataframe_empty_list():
    gdf = detections_to_geodataframe([], crs="EPSG:4326")
    assert len(gdf) == 0


def test_detections_to_geodataframe_raises_without_geo():
    detection = _make_detection((0, 0))  # center_geo no asignado
    with pytest.raises(ValueError):
        detections_to_geodataframe([detection], crs="EPSG:4326")
