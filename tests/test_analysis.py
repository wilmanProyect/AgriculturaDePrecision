# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el componente de análisis espacial.
"""

import pytest
import geopandas as gpd
from shapely.geometry import Polygon, Point
from core.analysis.spatial_analysis import SpatialAnalysis
from core.exceptions import InvalidCRSError

def test_point_in_polygon():
    """Prueba que los puntos se asignen y cuenten correctamente en cada parcela."""
    # Dos parcelas cuadradas de 10x10 metros proyectadas en UTM
    p1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])      # Parcela A
    p2 = Polygon([(20, 0), (20, 10), (30, 10), (30, 0), (20, 0)])    # Parcela B
    
    parcels_gdf = gpd.GeoDataFrame({
        'name': ['Parcela A', 'Parcela B'],
        'geometry': [p1, p2]
    }, crs="EPSG:32630")
    
    # 5 puntos representando plantas
    pt1 = Point(5, 5)     # Dentro de Parcela A
    pt2 = Point(3, 3)     # Dentro de Parcela A
    pt3 = Point(25, 5)    # Dentro de Parcela B
    pt4 = Point(15, 5)    # Fuera de ambas
    pt5 = Point(-5, -5)   # Fuera de ambas
    
    points_gdf = gpd.GeoDataFrame({
        'id': [1, 2, 3, 4, 5],
        'geometry': [pt1, pt2, pt3, pt4, pt5]
    }, crs="EPSG:32630")
    
    result = SpatialAnalysis.point_in_polygon(points_gdf, parcels_gdf, 'name')
    
    assert result['Parcela A'] == 2
    assert result['Parcela B'] == 1
    
    # Prueba con DataFrame de puntos vacío
    empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32630")
    empty_result = SpatialAnalysis.point_in_polygon(empty_gdf, parcels_gdf, 'name')
    assert empty_result['Parcela A'] == 0
    assert empty_result['Parcela B'] == 0

def test_point_in_polygon_crs_mismatch():
    """Prueba que se lance error si los CRS no coinciden."""
    p1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    parcels_gdf = gpd.GeoDataFrame({'name': ['A'], 'geometry': [p1]}, crs="EPSG:32630")
    
    pt = Point(5, 5)
    points_gdf = gpd.GeoDataFrame({'id': [1], 'geometry': [pt]}, crs="EPSG:4326")
    
    with pytest.raises(InvalidCRSError):
        SpatialAnalysis.point_in_polygon(points_gdf, parcels_gdf)

def test_calculate_buffer():
    """Prueba la generación de buffers geométricos."""
    p1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    gdf = gpd.GeoDataFrame({'geometry': [p1]}, crs="EPSG:32630")
    
    buffered = SpatialAnalysis.calculate_buffer(gdf, 2.0)
    assert buffered.geometry[0].area > gdf.geometry[0].area

def test_overlay_intersection():
    """Prueba la intersección de superposición de capas."""
    p1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    # Polígono que se superpone a p1 en el área [5, 15] x [0, 10]
    p2 = Polygon([(5, 0), (5, 10), (15, 10), (15, 0), (5, 0)])
    
    gdf1 = gpd.GeoDataFrame({'id1': [1], 'geometry': [p1]}, crs="EPSG:32630")
    gdf2 = gpd.GeoDataFrame({'id2': [2], 'geometry': [p2]}, crs="EPSG:32630")
    
    intersected = SpatialAnalysis.overlay_intersection(gdf1, gdf2)
    assert len(intersected) == 1
    # El área del rectángulo de intersección de 5x10 es 50.0
    assert intersected.geometry[0].area == 50.0
