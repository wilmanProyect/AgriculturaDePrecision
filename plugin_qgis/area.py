"""Medición de polígonos sin dependencias del motor de IA."""
import math

from qgis.core import QgsDistanceArea, QgsProject, QgsWkbTypes


def measure_polygon_areas(layer, selected_only=False):
    if layer is None or not layer.isValid():
        raise ValueError("Selecciona una capa de polígonos válida.")
    if layer.geometryType() != QgsWkbTypes.GeometryType.PolygonGeometry:
        raise ValueError("La capa debe contener polígonos.")
    if not layer.crs().isValid():
        raise ValueError("La capa necesita un sistema de coordenadas definido.")
    if selected_only and not layer.selectedFeatureCount():
        raise ValueError("Selecciona al menos un polígono en el mapa.")
    measure = QgsDistanceArea()
    measure.setSourceCrs(layer.crs(), QgsProject.instance().transformContext())
    measure.setEllipsoid("WGS84")
    features = layer.getSelectedFeatures() if selected_only else layer.getFeatures()
    rows = []
    for feature in features:
        geom = feature.geometry()
        if geom.isNull() or geom.isEmpty() or not geom.isGeosValid():
            raise ValueError(f"El polígono {feature.id()} está vacío o tiene geometría inválida. Corrígelo antes de medir.")
        area = measure.measureArea(geom)
        if not math.isfinite(area) or area <= 0:
            raise ValueError(f"No se pudo medir el polígono {feature.id()}.")
        rows.append((feature.id(), area, area / 10000))
    if not rows:
        raise ValueError("La capa no contiene polígonos.")
    return rows
