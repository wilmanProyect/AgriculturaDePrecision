# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el componente de exportación de datos.
"""

import os
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from core.exports.export_manager import ExportManager
from core.exceptions import UnsupportedFormatError

@pytest.fixture
def temp_output_dir():
    """Crea un directorio temporal para las salidas de prueba y lo limpia al terminar."""
    d = os.path.join(os.path.dirname(__file__), "data", "temp_outputs")
    os.makedirs(d, exist_ok=True)
    yield d
    import shutil
    import time
    if os.path.exists(d):
        # Reintentar en Windows por posibles bloqueos de archivos en el recolector de basura
        for _ in range(5):
            try:
                shutil.rmtree(d)
                break
            except PermissionError:
                time.sleep(0.1)


def test_export_geodataframe_geojson(temp_output_dir):
    """Prueba la exportación de capas espaciales a GeoJSON."""
    pt = Point(1.0, 1.0)
    gdf = gpd.GeoDataFrame({'name': ['Punto1'], 'geometry': [pt]}, crs="EPSG:4326")
    
    out_path = os.path.join(temp_output_dir, "test.geojson")
    ExportManager.export_geodataframe(gdf, out_path)
    
    assert os.path.exists(out_path)
    # Volver a leer para validar integridad
    read_gdf = gpd.read_file(out_path)
    assert len(read_gdf) == 1
    assert read_gdf.crs.to_epsg() == 4326
    assert read_gdf.iloc[0]['name'] == 'Punto1'

def test_export_geodataframe_invalid(temp_output_dir):
    """Prueba que se lance error si el formato espacial no está soportado."""
    gdf = gpd.GeoDataFrame({'geometry': [Point(1, 1)]}, crs="EPSG:4326")
    with pytest.raises(UnsupportedFormatError):
        ExportManager.export_geodataframe(gdf, os.path.join(temp_output_dir, "test.txt"))

def test_export_tabular_data_csv(temp_output_dir):
    """Prueba la exportación de resultados a CSV."""
    data = {'Parcela A': 100, 'Parcela B': 200}
    out_path = os.path.join(temp_output_dir, "test.csv")
    
    ExportManager.export_tabular_data(data, out_path, columns=['Parcela', 'Plantas'])
    
    assert os.path.exists(out_path)
    df = pd.read_csv(out_path)
    assert len(df) == 2
    assert list(df['Parcela']) == ['Parcela A', 'Parcela B']
    assert list(df['Plantas']) == [100, 200]

def test_export_tabular_data_xlsx(temp_output_dir):
    """Prueba la exportación de resultados a Excel (.xlsx)."""
    data = {'Parcela A': 100, 'Parcela B': 200}
    out_path = os.path.join(temp_output_dir, "test.xlsx")
    
    ExportManager.export_tabular_data(data, out_path, columns=['Parcela', 'Plantas'])
    
    assert os.path.exists(out_path)
    df = pd.read_excel(out_path)
    assert len(df) == 2
    assert list(df['Parcela']) == ['Parcela A', 'Parcela B']
    assert list(df['Plantas']) == [100, 200]

def test_export_tabular_invalid(temp_output_dir):
    """Prueba que se rechacen formatos tabulares no soportados."""
    data = {'A': 1}
    with pytest.raises(UnsupportedFormatError):
        ExportManager.export_tabular_data(data, os.path.join(temp_output_dir, "test.pdf"))
