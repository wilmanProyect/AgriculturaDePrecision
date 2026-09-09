# -*- coding: utf-8 -*-
"""
Genera capas de demostración para visualizar en QGIS el pipeline
"detección -> conteo por parcela -> densidad -> coloreado", usando
las parcelas de prueba (tests/data/parcelas.gpkg) y plantas SIMULADAS
(aún no hay un modelo YOLO11 entrenado con clases propias).

Salida (data/outputs/demo_qgis/):
    parcelas_analisis.gpkg   -> parcelas + num_plantas + densidad + clase
    plantas_detectadas.gpkg  -> puntos de plantas simuladas
    conteo_por_parcela.csv   -> resumen tabular
"""

import os
import random

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from core.analysis.spatial_analysis import SpatialAnalysis
from core.exports.export_manager import ExportManager
from core.geometry.geometry_manager import GeometryManager
from core.geometry.geometry_validator import GeometryValidator
from core.gis.vector.vector_manager import VectorManager
from core.logger import get_logger

logger = get_logger(__name__)

random.seed(42)
np.random.seed(42)

# Número de plantas simuladas por parcela (fijo, solo para que la demo se vea
# claramente diferenciada en Alta/Media/Baja densidad al abrirla en QGIS).
PLANT_COUNTS = {
    "Parcela A": 400,
    "Parcela B": 150,
    "Parcela C_invalida": 40,
}
DEFAULT_PLANT_COUNT = 100

OUTPUT_DIR = os.path.join("data", "outputs", "demo_qgis")


def _random_points_in_geometry(geom, n: int):
    """Genera n puntos aleatorios dentro de una geometría (Polygon o MultiPolygon) por rechazo."""
    minx, miny, maxx, maxy = geom.bounds
    points = []
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        candidate = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if geom.contains(candidate):
            points.append(candidate)
    if len(points) < n:
        logger.warning(f"Solo se generaron {len(points)}/{n} puntos dentro de la geometría.")
    return points


def _classify_density(gdf: gpd.GeoDataFrame, column: str) -> pd.Series:
    """Clasifica cada parcela en Alta/Media/Baja según su ranking de densidad."""
    ranking = gdf[column].rank(method="first", ascending=False)
    labels = pd.Series(index=gdf.index, dtype=object)
    n = len(gdf)
    for idx in gdf.index:
        percentile = ranking[idx] / n
        if percentile <= 1 / 3:
            labels[idx] = "Alta"
        elif percentile <= 2 / 3:
            labels[idx] = "Media"
        else:
            labels[idx] = "Baja"
    return labels


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with VectorManager() as vm:
        vm.open(os.path.join("tests", "data", "parcelas.gpkg"))
        parcelas = vm.get_features().copy()

    # Reparar geometrías inválidas (ej. la parcela "corbatín" usada en los tests)
    parcelas["geometry"] = parcelas["geometry"].apply(
        lambda g: g if GeometryValidator.is_valid(g) else GeometryValidator.repair(g)
    )

    crs_str = parcelas.crs.to_string()

    all_points, all_confidence = [], []
    area_by_parcel = {}

    for _, row in parcelas.iterrows():
        name = row["name"]
        area_m2 = GeometryManager.calculate_area(row.geometry, crs_str)
        area_by_parcel[name] = area_m2

        n_plants = PLANT_COUNTS.get(name, DEFAULT_PLANT_COUNT)
        points = _random_points_in_geometry(row.geometry, n_plants)
        all_points.extend(points)
        all_confidence.extend(np.random.uniform(0.6, 0.98, len(points)).tolist())

    plantas_gdf = gpd.GeoDataFrame(
        {
            "class_name": ["planta"] * len(all_points),
            "confidence": all_confidence
        },
        geometry=all_points,
        crs=crs_str
    )

    counts = SpatialAnalysis.point_in_polygon(plantas_gdf, parcelas, polygon_id_col="name")

    parcelas["area_m2"] = parcelas["name"].map(area_by_parcel)
    parcelas["num_plantas"] = parcelas["name"].map(counts).fillna(0).astype(int)
    parcelas["densidad_m2"] = parcelas["num_plantas"] / parcelas["area_m2"]
    parcelas["densidad_clase"] = _classify_density(parcelas, "densidad_m2")

    parcelas_out = os.path.join(OUTPUT_DIR, "parcelas_analisis.gpkg")
    plantas_out = os.path.join(OUTPUT_DIR, "plantas_detectadas.gpkg")
    csv_out = os.path.join(OUTPUT_DIR, "conteo_por_parcela.csv")

    ExportManager.export_geodataframe(parcelas, parcelas_out)
    ExportManager.export_geodataframe(plantas_gdf, plantas_out)
    ExportManager.export_tabular_data(counts, csv_out, columns=["Parcela", "Num_Plantas"])

    print("Capas de demostración generadas:")
    print(f"  - {parcelas_out} (parcelas + densidad + clase)")
    print(f"  - {plantas_out} ({len(plantas_gdf)} plantas simuladas)")
    print(f"  - {csv_out}")
    print("\nConteo por parcela:", counts)
    print(parcelas[["name", "area_m2", "num_plantas", "densidad_m2", "densidad_clase"]].to_string(index=False))


if __name__ == "__main__":
    main()
