"""Spatial acceptance tests: known rows/gaps, NoData and clipping."""
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from core.analysis.row_detection import (
    RowDetector, RowOptions, RowAnalysisCanceled, native_resolution_m, resolve_resolution_m,
    split_large_geometries
)


def scene(tmp_path, gap=True, nodata=False):
    height, width = 160, 320
    yy, xx = np.indices((height, width))
    vegetation = np.zeros((height, width), bool)
    for row in range(20, 141, 20):
        vegetation |= np.abs(yy-row) <= 2
    if gap:
        vegetation[58:63, 120:160] = False
    rgb = np.full((3, height, width), 100, dtype=np.uint8)
    rgb[1, vegetation] = 180
    rgb[0, vegetation] = 65
    rgb[2, vegetation] = 55
    if nodata:
        rgb[:, :, 120:160] = 0
    path = tmp_path/'scene.tif'
    transform = from_origin(500000, 8200000, .05, .05)
    with rasterio.open(path, 'w', driver='GTiff', width=width, height=height, count=3,
                       dtype='uint8', crs='EPSG:32720', transform=transform, nodata=0) as dst:
        dst.write(rgb)
    polygon = box(500000.1, 8199992.1, 500015.9, 8199999.9)
    parcels = gpd.GeoDataFrame({'id': [1]}, geometry=[polygon], crs=32720)
    return path, parcels


def test_known_rows_and_internal_gap(tmp_path):
    path, parcels = scene(tmp_path)
    result = RowDetector(RowOptions(spacing_m=1)).analyze(path, parcels)
    rows, gaps = result['rows_gdf'], result['gaps_gdf']
    assert 6 <= len(rows) <= 8
    assert len(gaps) == 1
    assert 1.4 <= gaps.iloc[0].long_m <= 2.2
    assert rows.is_valid.all() and gaps.is_valid.all()
    assert rows.difference(parcels.geometry.iloc[0].buffer(1e-6)).is_empty.all()
    assert gaps.iloc[0].geometry.distance(rows[rows.surco_id == gaps.iloc[0].surco_id].geometry.iloc[0]) < 1e-6


def test_analyze_accepts_precomputed_crs_and_skips_estimate_utm_crs(tmp_path, monkeypatch):
    path, parcels = scene(tmp_path)
    crs = parcels.estimate_utm_crs()
    metric = parcels.to_crs(crs)

    def _boom(*args, **kwargs):
        raise AssertionError("no debería llamarse estimate_utm_crs cuando se pasa crs=")
    monkeypatch.setattr(gpd.GeoDataFrame, "estimate_utm_crs", _boom)

    result = RowDetector(RowOptions(spacing_m=1)).analyze(path, metric, crs=crs)
    assert not result['rows_gdf'].empty


def test_nodata_is_not_a_sowing_gap(tmp_path):
    path, parcels = scene(tmp_path, gap=False, nodata=True)
    result = RowDetector(RowOptions(spacing_m=1, min_row_m=3)).analyze(path, parcels)
    assert not result['rows_gdf'].empty
    assert result['gaps_gdf'].empty


def test_gap_threshold_and_no_vegetation(tmp_path):
    path, parcels = scene(tmp_path)
    result = RowDetector(RowOptions(spacing_m=1, min_gap_m=3)).analyze(path, parcels)
    assert result['gaps_gdf'].empty
    with rasterio.open(path, 'r+') as dst:
        dst.write(np.full((3, dst.height, dst.width), 100, dtype='uint8'))
    result = RowDetector().analyze(path, parcels)
    assert result['rows_gdf'].empty
    assert result['warnings']


def test_native_resolution_m_matches_raster_pixel_size(tmp_path):
    path, parcels = scene(tmp_path)
    assert native_resolution_m(path, parcels.crs) == pytest.approx(0.05, abs=1e-6)


def test_resolve_resolution_m_keeps_explicit_value(tmp_path):
    path, parcels = scene(tmp_path)
    options = RowOptions(resolution_m=0.2)
    resolved = resolve_resolution_m(options, path, parcels)
    assert resolved.resolution_m == 0.2


def test_resolve_resolution_m_auto_matches_native_with_generous_budget(tmp_path):
    path, parcels = scene(tmp_path)
    options = RowOptions(resolution_m=None, max_pixels=50_000_000)
    resolved = resolve_resolution_m(options, path, parcels)
    assert resolved.resolution_m == pytest.approx(0.05, abs=1e-6)


def test_resolve_resolution_m_auto_ignores_max_pixels(tmp_path):
    """Ya no achica la resolución según el área: lotes grandes se dividen en piezas
    (`split_large_geometries`), no se pierde detalle real coarseando la resolución."""
    path, parcels = scene(tmp_path)
    options = RowOptions(resolution_m=None, max_pixels=100)
    resolved = resolve_resolution_m(options, path, parcels)
    assert resolved.resolution_m == pytest.approx(0.05, abs=1e-6)


def test_split_large_geometries_keeps_small_geometries_intact():
    parcels = gpd.GeoDataFrame({'id': [1]}, geometry=[box(0, 0, 5, 5)], crs=32720)
    pieces = split_large_geometries(parcels, max_area_m2=100)
    assert pieces == [("1", parcels.geometry.iloc[0])]


def test_split_large_geometries_splits_and_conserves_area():
    big = box(0, 0, 100, 60)  # 6000 m²
    parcels = gpd.GeoDataFrame({'id': [1]}, geometry=[big], crs=32720)
    pieces = split_large_geometries(parcels, max_area_m2=100)  # tiles de 10x10 m
    assert len(pieces) > 1
    assert all(geom.area <= 100 + 1e-6 for _, geom in pieces)
    assert all(label.startswith("1.") for label, _ in pieces)
    assert sum(geom.area for _, geom in pieces) == pytest.approx(big.area, rel=1e-6)


def test_analyze_handles_large_parcel_via_tiling_without_raising(tmp_path):
    """Antes de la división en piezas, un max_pixels tan chico para esta parcela
    tiraba 'Parcela demasiado grande'. Ahora se resuelve dividiendo en piezas
    (max_pixels=15000 fuerza varias piezas de ~5x4 m; min_row_m se baja de su
    default de 5 m porque si no, ninguna hilera partida por la grilla lo supera)."""
    path, parcels = scene(tmp_path)
    result = RowDetector(RowOptions(spacing_m=1, resolution_m=.05, max_pixels=15000, min_row_m=1.0)).analyze(
        path, parcels)
    assert not result['rows_gdf'].empty
    assert any('.' in pid for pid in result['rows_gdf']['parcela_id'])


def test_analyze_with_default_auto_resolution_still_detects_rows_and_gap(tmp_path):
    path, parcels = scene(tmp_path)
    result = RowDetector(RowOptions(spacing_m=1)).analyze(path, parcels)
    assert not result['rows_gdf'].empty
    assert len(result['gaps_gdf']) == 1


def test_cancellation_and_options(tmp_path):
    path, parcels = scene(tmp_path)
    with pytest.raises(RowAnalysisCanceled):
        RowDetector().analyze(path, parcels, should_stop=lambda: True)
    with pytest.raises(ValueError):
        RowDetector(RowOptions(spacing_m=.1, resolution_m=.05))
    # max_pixels chico ya no falla: la parcela se divide en piezas (ver
    # test_analyze_handles_large_parcel_via_tiling_without_raising).
    result = RowDetector(RowOptions(resolution_m=.05, max_pixels=5000)).analyze(path, parcels)
    assert isinstance(result['rows_gdf'], gpd.GeoDataFrame)


def test_two_sowing_directions():
    yy, xx = np.indices((240, 640))
    signal = np.zeros((240, 640), dtype='float32')
    for y in range(20, 221, 20):
        signal[(np.abs(yy-y) <= 2) & (xx < 390)] = .4
    for x in range(420, 621, 20):
        signal[(np.abs(xx-x) <= 2)] = .4
    valid = np.ones_like(signal, dtype=bool)
    transform = from_origin(500000, 8200000, .05, .05)
    rows, gaps = RowDetector(RowOptions(spacing_m=1, resolution_m=.05)).trace_signal(
        signal, valid, transform, box(500000, 8199988, 500032, 8200000))
    assert len({r['sector'] for r in rows}) == 2
    horizontal = [r for r in rows if r['geometry'].bounds[2]-r['geometry'].bounds[0] > 10]
    vertical = [r for r in rows if r['geometry'].bounds[3]-r['geometry'].bounds[1] > 8]
    assert len(horizontal) >= 9 and len(vertical) >= 9
    assert not gaps


def test_oblique_rows():
    from scipy.ndimage import rotate
    yy, xx = np.indices((200, 320))
    signal = np.zeros((200, 320), dtype='float32')
    for y in range(20, 181, 20):
        signal[np.abs(yy-y) <= 2] = .4
    rotated = rotate(signal, 27, reshape=True, order=1)
    valid = rotate(np.ones_like(signal), 27, reshape=True, order=0) > .5
    h, w = rotated.shape
    rows, _ = RowDetector(RowOptions(spacing_m=1, resolution_m=.05)).trace_signal(
        rotated, valid, from_origin(500000, 8200000, .05, .05),
        box(500000, 8200000-h*.05, 500000+w*.05, 8200000))
    assert 8 <= len(rows) <= 10
    for row in rows:
        points = np.array(row['geometry'].coords)
        angle = np.rad2deg(np.arctan2(*(points[-1]-points[0])[::-1])) % 180
        assert abs(angle-27) < 3
