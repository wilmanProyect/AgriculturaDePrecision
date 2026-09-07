# -*- coding: utf-8 -*-
"""
Pruebas unitarias para las funciones de validación y mediciones geométricas.
"""

import pytest
from shapely.geometry import Polygon, Point
from core.geometry.geometry_validator import GeometryValidator
from core.geometry.geometry_manager import GeometryManager
from core.exceptions import InvalidGeometryError

def test_geometry_validation():
    """Prueba la detección de geometrías válidas e inválidas."""
    # Polígono cuadrado estándar (válido)
    p_val = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    assert GeometryValidator.is_valid(p_val) is True
    
    # Polígono auto-intersecante tipo corbatín (inválido)
    p_inval = Polygon([(0, 0), (0, 2), (2, 0), (2, 2), (0, 0)])
    assert GeometryValidator.is_valid(p_inval) is False
    assert GeometryValidator.is_valid(None) is False

def test_geometry_repair():
    """Prueba que se reparen correctamente las geometrías inválidas."""
    p_inval = Polygon([(0, 0), (0, 2), (2, 0), (2, 2), (0, 0)])
    repaired = GeometryValidator.repair(p_inval)
    assert GeometryValidator.is_valid(repaired) is True
    
    # Un polígono que ya es válido no debe sufrir alteraciones
    p_val = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    repaired_val = GeometryValidator.repair(p_val)
    assert repaired_val == p_val

    with pytest.raises(InvalidGeometryError):
        GeometryValidator.repair(None)

def test_geometry_calculations_projected():
    """Prueba mediciones en un CRS proyectado métrico."""
    p_val = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    area = GeometryManager.calculate_area(p_val, "EPSG:32630")
    perim = GeometryManager.calculate_perimeter(p_val, "EPSG:32630")
    
    assert area == 100.0
    assert perim == 40.0
    
    cx, cy = GeometryManager.calculate_centroid(p_val)
    assert cx == 5.0
    assert cy == 5.0

def test_geometry_calculations_geographic():
    """Prueba mediciones sobre un CRS geográfico (grados decimales)."""
    # Polígono alrededor de Madrid en WGS84
    p_geo = Polygon([(-3.95, 40.45), (-3.90, 40.45), (-3.90, 40.40), (-3.95, 40.40), (-3.95, 40.45)])
    area = GeometryManager.calculate_area(p_geo, "EPSG:4326")
    perim = GeometryManager.calculate_perimeter(p_geo, "EPSG:4326")
    
    # El área calculada debe estar en metros cuadrados (no en grados cuadrados)
    # 0.05 grados es aprox. 4.2 x 5.5 km (~23 millones de m2)
    assert area > 20000000.0
    assert perim > 15000.0

def test_geometry_reprojection():
    """Prueba que las geometrías se proyecten al huso UTM correcto."""
    pt = Point(-3.70379, 40.41678)  # Coordenadas de Madrid
    utm_crs = GeometryManager.get_utm_crs_for_point(pt.x, pt.y)
    assert utm_crs == "EPSG:32630"  # Madrid está en el huso 30N (hemisferio norte)
    
    pt_proj = GeometryManager.project_geometry(pt, "EPSG:4326", utm_crs)
    assert pt_proj.x != pt.x
    assert pt_proj.y != pt.y
