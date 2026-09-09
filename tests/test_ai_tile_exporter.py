# -*- coding: utf-8 -*-
"""
Pruebas unitarias para TileExporter (exportación de tiles para anotación).
"""

import os

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from ai.training.tile_exporter import TileExporter
from core.gis.raster.raster_manager import RasterManager


@pytest.fixture
def uniform_raster(tmp_path):
    """Ráster de 20x20 (1 banda) con valor constante y sin NoData."""
    raster_path = tmp_path / "uniforme.tif"
    transform = from_origin(-4.0, 40.5, 0.01, 0.01)
    with rasterio.open(
        str(raster_path), "w", driver="GTiff", height=20, width=20, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(np.full((20, 20), 100, dtype="uint8"), 1)
    return str(raster_path)


@pytest.fixture
def raster_with_nodata(tmp_path):
    """Ráster de 20x20 donde la mitad derecha es NoData (simula borde de vuelo)."""
    raster_path = tmp_path / "con_nodata.tif"
    transform = from_origin(-4.0, 40.5, 0.01, 0.01)
    data = np.full((20, 20), 100, dtype="uint8")
    data[:, 10:] = 0  # mitad derecha = NoData
    with rasterio.open(
        str(raster_path), "w", driver="GTiff", height=20, width=20, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0
    ) as dst:
        dst.write(data, 1)
    return str(raster_path)


def test_export_tiles_writes_expected_files(uniform_raster, tmp_path):
    output_dir = str(tmp_path / "tiles")
    exporter = TileExporter(tile_size=10, overlap=0.0)

    with RasterManager() as rm:
        rm.open(uniform_raster)
        written = exporter.export_tiles(rm, output_dir, prefix="demo")

    assert len(written) == 4  # 20x20 -> grilla 2x2 de tiles de 10x10
    for path in written:
        assert os.path.exists(path)
        img = Image.open(path)
        assert img.size == (10, 10)
        assert img.mode == "RGB"


def test_export_tiles_skips_mostly_nodata_tiles(raster_with_nodata, tmp_path):
    output_dir = str(tmp_path / "tiles")
    exporter = TileExporter(tile_size=10, overlap=0.0, min_valid_fraction=0.5)

    with RasterManager() as rm:
        rm.open(raster_with_nodata)
        written = exporter.export_tiles(rm, output_dir, prefix="demo")

    # Solo la columna izquierda (2 tiles) tiene datos validos; la derecha es NoData
    assert len(written) == 2


def test_export_tiles_respects_bounds(uniform_raster, tmp_path):
    output_dir = str(tmp_path / "tiles")
    exporter = TileExporter(tile_size=10, overlap=0.0)

    with RasterManager() as rm:
        rm.open(uniform_raster)
        # Limitar al cuadrante superior izquierdo -> solo 1 tile
        written = exporter.export_tiles(rm, output_dir, bounds=(-4.0, 40.4, -3.9, 40.5))

    assert len(written) == 1


def test_export_tiles_requires_open_raster(tmp_path):
    exporter = TileExporter()
    with pytest.raises(RuntimeError):
        exporter.export_tiles(RasterManager(), str(tmp_path / "tiles"))


def test_to_uint8_scales_float_data():
    image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    # Replicar a un array con suficientes valores para percentiles significativos
    image = np.tile(image, (10, 10, 1))
    result = TileExporter._to_uint8(image)
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255


def test_to_uint8_handles_constant_image():
    image = np.full((5, 5, 3), 42.0, dtype=np.float32)
    result = TileExporter._to_uint8(image)
    assert result.dtype == np.uint8
    assert np.all(result == 0)
