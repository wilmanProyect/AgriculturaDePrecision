# -*- coding: utf-8 -*-
"""
Script para generar datos de prueba espaciales (.tif, .shp, .gpkg)
para la suite de pruebas unitarias.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Polygon

def generate_test_raster(output_path: str) -> None:
    """Genera un archivo GeoTIFF de prueba de 100x100 píxeles."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 100x100 píxeles, resolución de 0.5 grados por píxel centrados en España
    width, height = 100, 100
    transform = from_origin(-4.0, 40.5, 0.01, 0.01)
    
    # Escribir GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='uint8',
        crs='EPSG:4326',
        transform=transform,
        nodata=0
    ) as dst:
        # Rellenar con un valor constante (100)
        data = np.ones((height, width), dtype='uint8') * 100
        # Poner una región con valor NoData (0) en el centro para probar estadísticas
        data[40:60, 40:60] = 0
        dst.write(data, 1)

def generate_test_vectors(shp_path: str, gpkg_path: str) -> None:
    """Genera capas vectoriales de prueba (Shapefile y GeoPackage)."""
    os.makedirs(os.path.dirname(os.path.abspath(shp_path)), exist_ok=True)
    
    # Crear 3 polígonos representando parcelas agrícolas
    p1 = Polygon([(-3.95, 40.45), (-3.90, 40.45), (-3.90, 40.40), (-3.95, 40.40), (-3.95, 40.45)])
    p2 = Polygon([(-3.89, 40.45), (-3.84, 40.45), (-3.84, 40.40), (-3.89, 40.40), (-3.89, 40.45)])
    
    # Polígono auto-intersecante (inválido) en forma de corbatín para probar reparaciones
    p3 = Polygon([(-3.95, 40.39), (-3.90, 40.37), (-3.95, 40.37), (-3.90, 40.39), (-3.95, 40.39)])
    
    gdf = gpd.GeoDataFrame({
        'name': ['Parcela A', 'Parcela B', 'Parcela C_invalida'],
        'geometry': [p1, p2, p3]
    }, crs='EPSG:4326')
    
    # Guardar a Shapefile y GeoPackage
    gdf.to_file(shp_path)
    gdf.to_file(gpkg_path, driver='GPKG')

def main():
    print("Iniciando generación de datos de prueba...")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    raster_path = os.path.join(data_dir, "ortomosaico.tif")
    shp_path = os.path.join(data_dir, "parcelas.shp")
    gpkg_path = os.path.join(data_dir, "parcelas.gpkg")
    
    generate_test_raster(raster_path)
    print(f"Ráster de prueba creado en: {raster_path}")
    
    generate_test_vectors(shp_path, gpkg_path)
    print(f"Vectores (SHP/GPKG) de prueba creados en: {shp_path} y {gpkg_path}")
    print("Datos de prueba creados correctamente.")

if __name__ == "__main__":
    main()
