# -*- coding: utf-8 -*-
"""
Conversión entre tipos de QGIS (QgsVectorLayer/QgsFeature) y GeoDataFrame
de GeoPandas. Esta es la única capa que conoce tanto QGIS como el Motor GIS
(core) / Motor IA (ai): mantiene la regla del proyecto de no mezclar lógica
de QGIS dentro de core/ai.
"""

import os
from typing import Optional

import geopandas as gpd
from shapely import wkb

from qgis.core import QgsFeature, QgsField, QgsFields, QgsGeometry, QgsProject, QgsVectorLayer
from qgis.PyQt.QtCore import QVariant

_DTYPE_TO_QVARIANT = {
    "int64": QVariant.Int,
    "int32": QVariant.Int,
    "float64": QVariant.Double,
    "float32": QVariant.Double,
    "bool": QVariant.Bool,
    "object": QVariant.String,
}

_GEOM_TYPE_TO_MEMORY_PROVIDER = {
    "Point": "Point",
    "MultiPoint": "MultiPoint",
    "LineString": "LineString",
    "MultiLineString": "MultiLineString",
    "Polygon": "Polygon",
    "MultiPolygon": "MultiPolygon",
}


def layer_source_path(layer) -> Optional[str]:
    """
    Extrae la ruta de archivo de una capa de QGIS, ignorando sufijos como
    '|layername=...' que usa GDAL/OGR para capas dentro de un GeoPackage.
    Devuelve None si la capa no está respaldada por un archivo existente
    (por ejemplo, una capa en memoria).
    """
    if layer is None:
        return None
    path = layer.source().split('|')[0]
    return path if os.path.exists(path) else None


def qgs_vector_layer_to_geodataframe(layer: QgsVectorLayer) -> gpd.GeoDataFrame:
    """Convierte una QgsVectorLayer (de archivo o en memoria) a un GeoDataFrame."""
    field_names = [f.name() for f in layer.fields()]
    records, geometries = [], []

    for feature in layer.getFeatures():
        geom = feature.geometry()
        if geom is None or geom.isEmpty():
            continue
        geometries.append(wkb.loads(bytes(geom.asWkb())))
        records.append({name: feature[name] for name in field_names})

    crs = layer.crs().authid() or None
    return gpd.GeoDataFrame(records, geometry=geometries, crs=crs)


def geodataframe_to_qgs_layer(gdf: gpd.GeoDataFrame, name: str) -> QgsVectorLayer:
    """
    Convierte un GeoDataFrame a una QgsVectorLayer en memoria, la añade
    al proyecto activo de QGIS y la devuelve.
    """
    crs = gdf.crs.to_string() if gdf.crs is not None else "EPSG:4326"
    geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else "Point"
    provider_geom_type = _GEOM_TYPE_TO_MEMORY_PROVIDER.get(geom_type, "Point")

    layer = QgsVectorLayer(f"{provider_geom_type}?crs={crs}", name, "memory")
    provider = layer.dataProvider()

    attribute_columns = [c for c in gdf.columns if c != "geometry"]
    fields = QgsFields()
    for col in attribute_columns:
        dtype = str(gdf[col].dtype)
        fields.append(QgsField(col, _DTYPE_TO_QVARIANT.get(dtype, QVariant.String)))
    provider.addAttributes(fields)
    layer.updateFields()

    features = []
    for _, row in gdf.iterrows():
        feature = QgsFeature(layer.fields())
        # QgsGeometry.fromWkb() como método estático falla con el binding SIP/PyQt6
        # de QGIS 4.x ("first argument of unbound method must have type 'QgsGeometry'");
        # fromWkt() sí funciona de forma fiable como estático en ambas versiones.
        feature.setGeometry(QgsGeometry.fromWkt(row.geometry.wkt))
        feature.setAttributes([row[col] for col in attribute_columns])
        features.append(feature)

    if features:
        provider.addFeatures(features)
    layer.updateExtents()

    QgsProject.instance().addMapLayer(layer)
    return layer
