# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el cálculo de índices de vegetación y estadísticas zonales.
"""

import numpy as np
import pytest
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from core.analysis.vegetation_index import VegetationIndexCalculator, DEFAULT_NDVI_THRESHOLDS
from core.exceptions import RasterNotFoundError


def test_compute_ndvi_known_values():
    nir = np.array([[0.5]])
    red = np.array([[0.1]])
    ndvi = VegetationIndexCalculator.compute_ndvi(nir, red)
    assert ndvi[0, 0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1))


def test_compute_ndvi_handles_zero_denominator():
    nir = np.array([[0.0]])
    red = np.array([[0.0]])
    ndvi = VegetationIndexCalculator.compute_ndvi(nir, red)
    assert np.isnan(ndvi[0, 0])


def test_compute_ndre_known_values():
    nir = np.array([[0.6]])
    red_edge = np.array([[0.2]])
    ndre = VegetationIndexCalculator.compute_ndre(nir, red_edge)
    assert ndre[0, 0] == pytest.approx((0.6 - 0.2) / (0.6 + 0.2))


@pytest.mark.parametrize("value,expected", [
    (0.8, "Vigorosa"),
    (0.6, "Vigorosa"),
    (0.45, "Moderada"),
    (0.3, "Moderada"),
    (0.15, "Estresada"),
    (0.05, "Suelo/Agua"),
    (-0.2, "Suelo/Agua"),
])
def test_classify_value_thresholds(value, expected):
    assert VegetationIndexCalculator.classify_value(value, DEFAULT_NDVI_THRESHOLDS) == expected


def test_classify_value_handles_missing_data():
    assert VegetationIndexCalculator.classify_value(None) == "Sin datos"
    assert VegetationIndexCalculator.classify_value(float("nan")) == "Sin datos"


def test_zonal_statistics_raises_when_raster_missing(tmp_path):
    gdf = gpd.GeoDataFrame({"name": ["A"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32630")
    with pytest.raises(RasterNotFoundError):
        VegetationIndexCalculator.zonal_statistics(str(tmp_path / "no_existe.tif"), gdf)


def test_zonal_statistics_computes_correct_stats_and_class(tmp_path):
    raster_path = tmp_path / "ndvi.tif"
    width, height = 20, 20
    transform = from_origin(0, 20, 1, 1)

    data = np.zeros((height, width), dtype=np.float32)
    data[:, :10] = 0.7   # mitad izquierda: vegetación vigorosa
    data[:, 10:] = 0.2   # mitad derecha: vegetación estresada

    with rasterio.open(
        str(raster_path), "w", driver="GTiff",
        height=height, width=width, count=1, dtype="float32",
        crs="EPSG:32630", transform=transform, nodata=np.nan
    ) as dst:
        dst.write(data, 1)

    parcelas = gpd.GeoDataFrame(
        {"name": ["Parcela Vigorosa", "Parcela Estresada"]},
        geometry=[box(0, 0, 10, 20), box(10, 0, 20, 20)],
        crs="EPSG:32630"
    )

    result = VegetationIndexCalculator.zonal_statistics(str(raster_path), parcelas)

    vigorosa = result[result["name"] == "Parcela Vigorosa"].iloc[0]
    estresada = result[result["name"] == "Parcela Estresada"].iloc[0]

    assert vigorosa["ndvi_mean"] == pytest.approx(0.7, abs=1e-3)
    assert vigorosa["ndvi_clase"] == "Vigorosa"
    assert estresada["ndvi_mean"] == pytest.approx(0.2, abs=1e-3)
    assert estresada["ndvi_clase"] == "Estresada"
    assert result.crs.to_string() == "EPSG:32630"


def test_zonal_statistics_reprojects_parcels_to_raster_crs(tmp_path):
    raster_path = tmp_path / "ndvi_geo.tif"
    width, height = 10, 10
    transform = from_origin(-4.0, 40.5, 0.01, 0.01)
    data = np.full((height, width), 0.5, dtype=np.float32)

    with rasterio.open(
        str(raster_path), "w", driver="GTiff",
        height=height, width=width, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform, nodata=np.nan
    ) as dst:
        dst.write(data, 1)

    # Parcela definida en un CRS proyectado distinto al del ráster (EPSG:4326)
    parcela_wgs84 = box(-4.0, 40.4, -3.9, 40.5)
    parcelas = gpd.GeoDataFrame({"name": ["A"]}, geometry=[parcela_wgs84], crs="EPSG:4326").to_crs("EPSG:32630")

    result = VegetationIndexCalculator.zonal_statistics(str(raster_path), parcelas)

    assert result.iloc[0]["ndvi_mean"] == pytest.approx(0.5, abs=1e-3)
    assert result.crs.to_string() == "EPSG:32630"
