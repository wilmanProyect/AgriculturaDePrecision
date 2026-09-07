# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el componente de administración de capas vectoriales.
"""

import os
import pytest
from core.gis.vector.vector_manager import VectorManager
from core.exceptions import UnsupportedFormatError

@pytest.fixture
def shp_path():
    """Retorna la ruta al Shapefile de prueba."""
    return os.path.join(os.path.dirname(__file__), "data", "parcelas.shp")

@pytest.fixture
def gpkg_path():
    """Retorna la ruta al GeoPackage de prueba."""
    return os.path.join(os.path.dirname(__file__), "data", "parcelas.gpkg")

@pytest.fixture
def kml_path():
    """Retorna la ruta al archivo KML de prueba."""
    return os.path.join(os.path.dirname(__file__), "data", "parcelas.kml")

def test_vector_manager_open_close(shp_path):
    """Prueba que el VectorManager abra y cierre capas vectoriales."""
    manager = VectorManager()
    manager.open(shp_path)
    assert manager.gdf is not None
    assert manager.file_path == shp_path
    manager.close()
    assert manager.gdf is None
    assert manager.file_path is None

def test_vector_manager_not_found():
    """Prueba que se lance error si el archivo no existe."""
    manager = VectorManager()
    with pytest.raises(FileNotFoundError):
        manager.open("tests/data/vector_que_no_existe_987.shp")

def test_vector_manager_unsupported_format():
    """Prueba que se rechacen archivos con formatos inválidos."""
    manager = VectorManager()
    with pytest.raises(UnsupportedFormatError):
        manager.open(__file__)

def test_vector_metadata_shp(shp_path):
    """Prueba que se extraigan correctamente los metadatos de un Shapefile."""
    with VectorManager() as manager:
        manager.open(shp_path)
        meta = manager.get_metadata()
        
        assert meta.name == "parcelas.shp"
        assert meta.feature_count == 3
        assert "Polygon" in meta.geom_type
        assert "4326" in meta.crs
        assert len(meta.bounds) == 4
        assert "name" in meta.fields
        # El área y perímetro deben calcularse de forma métrica (proyección UTM implícita)
        assert meta.total_area > 0.0
        assert meta.total_perimeter > 0.0
        
        # Test de serialización a dict
        meta_dict = meta.to_dict()
        assert meta_dict['feature_count'] == 3
        assert meta_dict['crs'] == meta.crs

def test_vector_metadata_gpkg(gpkg_path):
    """Prueba que se extraigan los metadatos de un GeoPackage."""
    with VectorManager() as manager:
        manager.open(gpkg_path)
        meta = manager.get_metadata()
        
        assert meta.name == "parcelas.gpkg"
        assert meta.feature_count == 3
        assert meta.total_area > 0.0

def test_vector_kml_parsing(kml_path):
    """Prueba que se puedan importar y leer archivos KML (ya sea nativamente o con el fallback)."""
    with VectorManager() as manager:
        manager.open(kml_path)
        gdf = manager.get_features()
        
        assert gdf is not None
        assert len(gdf) == 2
        names = list(gdf['name'])
        assert "Parcela A" in names
        assert "Parcela B" in names

def test_vector_kml_fallback_parser(kml_path):
    """Prueba directamente la función de fallback del lector de KML."""
    from core.gis.vector.vector_utils import parse_kml_fallback
    gdf = parse_kml_fallback(kml_path)
    assert gdf is not None
    assert len(gdf) == 2
    assert list(gdf['name']) == ["Parcela A", "Parcela B"]

