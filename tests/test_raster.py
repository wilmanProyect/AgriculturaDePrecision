# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el componente de administración de rásteres.
"""

import os
import pytest
import numpy as np
from core.gis.raster.raster_manager import RasterManager
from core.exceptions import RasterNotFoundError, UnsupportedFormatError

@pytest.fixture
def raster_path():
    """Retorna la ruta al GeoTIFF de prueba."""
    return os.path.join(os.path.dirname(__file__), "data", "ortomosaico.tif")

def test_raster_manager_open_close(raster_path):
    """Prueba la apertura y cierre explícito de un ráster."""
    manager = RasterManager()
    manager.open(raster_path)
    assert manager.dataset is not None
    assert manager.file_path == raster_path
    manager.close()
    assert manager.dataset is None
    assert manager.file_path is None

def test_raster_manager_context_manager(raster_path):
    """Prueba el uso de RasterManager como context manager."""
    with RasterManager() as manager:
        manager.open(raster_path)
        assert manager.dataset is not None
    assert manager.dataset is None

def test_raster_manager_not_found():
    """Prueba la excepción cuando el archivo no existe."""
    manager = RasterManager()
    with pytest.raises(RasterNotFoundError):
        manager.open("tests/data/archivo_inexistente_123.tif")

def test_raster_manager_invalid_format():
    """Prueba que se rechacen formatos no soportados."""
    manager = RasterManager()
    with pytest.raises(UnsupportedFormatError):
        # Intentar abrir un archivo de python como si fuera ráster
        manager.open(__file__)

def test_raster_metadata(raster_path):
    """Prueba la extracción correcta de metadatos."""
    with RasterManager() as manager:
        manager.open(raster_path)
        meta = manager.get_metadata()
        
        assert meta.name == "ortomosaico.tif"
        assert meta.width == 100
        assert meta.height == 100
        assert meta.count == 1
        assert "4326" in meta.crs
        assert round(meta.res_x, 4) == 0.01
        assert round(meta.res_y, 4) == 0.01
        assert len(meta.bounds) == 4
        assert meta.dtype == "uint8"
        assert meta.nodata == 0
        assert meta.size_bytes > 0
        
        # Test conversión a dict
        meta_dict = meta.to_dict()
        assert isinstance(meta_dict, dict)
        assert meta_dict['width'] == 100
        assert meta_dict['crs'] == meta.crs

def test_raster_read_band(raster_path):
    """Prueba la lectura de matrices de píxeles."""
    with RasterManager() as manager:
        manager.open(raster_path)
        band_data = manager.read_band(1)
        
        assert isinstance(band_data, np.ndarray)
        assert band_data.shape == (100, 100)
        # El centro tiene el valor NoData asignado (0)
        assert band_data[50, 50] == 0
        # Las esquinas tienen el valor de datos (100)
        assert band_data[0, 0] == 100

def test_raster_statistics(raster_path):
    """Prueba que el cálculo de estadísticas filtre correctamente el NoData."""
    with RasterManager() as manager:
        manager.open(raster_path)
        stats = manager.get_statistics(1)
        
        # Como excluimos el valor NoData (0), todos los píxeles restantes son 100
        assert stats['min'] == 100.0
        assert stats['max'] == 100.0
        assert stats['mean'] == 100.0
        assert stats['median'] == 100.0
        assert stats['std'] == 0.0

def test_raster_invalid_band_index(raster_path):
    """Prueba que se controle el acceso a índices de banda inválidos."""
    with RasterManager() as manager:
        manager.open(raster_path)
        with pytest.raises(ValueError):
            manager.read_band(2)  # El ráster de prueba solo tiene 1 banda

def test_raster_coordinate_transform(raster_path):
    """Prueba las utilidades de transformación de píxeles a coordenadas y viceversa."""
    from core.gis.raster.raster_utils import pixel_to_coords, coords_to_pixel
    with RasterManager() as manager:
        manager.open(raster_path)
        meta = manager.get_metadata()
        
        # Probar transformación de píxel a coordenadas (retorna el centro del píxel)
        x, y = pixel_to_coords(0, 0, meta.transform)
        # Debe coincidir con el límite superior izquierdo de la Bounding Box más la mitad del tamaño del píxel
        assert round(x, 4) == round(meta.bounds[0] + meta.res_x / 2, 4)
        assert round(y, 4) == round(meta.bounds[3] - meta.res_y / 2, 4)
        
        # Probar transformación inversa de coordenadas a píxel
        row, col = coords_to_pixel(x, y, meta.transform)
        assert row == 0
        assert col == 0


