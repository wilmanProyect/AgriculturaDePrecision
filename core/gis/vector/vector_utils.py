# -*- coding: utf-8 -*-
"""
Funciones auxiliares para manipulación y transformación de datos vectoriales.
"""

import re
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point

def parse_kml_fallback(file_path: str) -> gpd.GeoDataFrame:
    """
    Parser KML alternativo que lee directamente el XML del archivo KML y
    construye un GeoDataFrame de geopandas con geometría y atributos.
    Se utiliza como alternativa si Fiona no soporta el driver KML.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Error al analizar el XML del archivo KML: {e}")

    features = []
    
    # Buscar todos los elementos Placemark usando búsqueda recursiva
    # Consideramos namespaces comunes de KML 2.2
    placemarks = root.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
    if not placemarks:
        placemarks = root.findall('.//Placemark')

    for placemark in placemarks:
        # Obtener nombre si existe
        name_el = placemark.find('{http://www.opengis.net/kml/2.2}name')
        if name_el is None:
            name_el = placemark.find('name')
        name = name_el.text.strip() if (name_el is not None and name_el.text) else "Placemark"
        
        geom = None
        
        # 1. Intentar buscar Polígono
        poly_el = placemark.find('.//{http://www.opengis.net/kml/2.2}Polygon')
        if poly_el is None:
            poly_el = placemark.find('.//Polygon')
            
        if poly_el is not None:
            coord_el = poly_el.find('.//{http://www.opengis.net/kml/2.2}coordinates')
            if coord_el is None:
                coord_el = poly_el.find('.//coordinates')
            if coord_el is not None and coord_el.text:
                coords_str = coord_el.text.strip()
                coords = []
                # El formato en KML es lon,lat,alt separados por espacios
                for pt in coords_str.split():
                    parts = pt.split(',')
                    if len(parts) >= 2:
                        coords.append((float(parts[0]), float(parts[1])))
                if len(coords) >= 3:
                    geom = Polygon(coords)
                    
        # 2. Intentar buscar Línea
        if geom is None:
            line_el = placemark.find('.//{http://www.opengis.net/kml/2.2}LineString')
            if line_el is None:
                line_el = placemark.find('.//LineString')
            if line_el is not None:
                coord_el = line_el.find('.//{http://www.opengis.net/kml/2.2}coordinates')
                if coord_el is None:
                    coord_el = line_el.find('.//coordinates')
                if coord_el is not None and coord_el.text:
                    coords_str = coord_el.text.strip()
                    coords = []
                    for pt in coords_str.split():
                        parts = pt.split(',')
                        if len(parts) >= 2:
                            coords.append((float(parts[0]), float(parts[1])))
                    if len(coords) >= 2:
                        geom = LineString(coords)
                        
        # 3. Intentar buscar Punto
        if geom is None:
            pt_el = placemark.find('.//{http://www.opengis.net/kml/2.2}Point')
            if pt_el is None:
                pt_el = placemark.find('.//Point')
            if pt_el is not None:
                coord_el = pt_el.find('.//{http://www.opengis.net/kml/2.2}coordinates')
                if coord_el is None:
                    coord_el = pt_el.find('.//coordinates')
                if coord_el is not None and coord_el.text:
                    parts = coord_el.text.strip().split(',')
                    if len(parts) >= 2:
                        geom = Point(float(parts[0]), float(parts[1]))

        if geom is not None:
            features.append({
                'name': name,
                'geometry': geom
            })

    if not features:
        return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")

    df = pd.DataFrame(features)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf
