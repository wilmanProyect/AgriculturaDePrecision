# -*- coding: utf-8 -*-
"""
Tarea en segundo plano (QgsTask) que ejecuta la detección YOLO11 y el análisis
espacial sin bloquear la interfaz de QGIS.
"""

import geopandas as gpd
import pandas as pd

from qgis.core import QgsTask

from ai.inference.detector import PlantDetector
from ai.utils.georeferencing import detections_to_geodataframe
from core.analysis.spatial_analysis import SpatialAnalysis
from core.analysis.vegetation_index import VegetationIndexCalculator
from core.geometry.geometry_manager import GeometryManager
from core.geometry.geometry_validator import GeometryValidator
from core.gis.raster.raster_manager import RasterManager
from core.logger import get_logger

logger = get_logger(__name__)


class DetectionTask(QgsTask):
    """
    Ejecuta, fuera del hilo principal de QGIS: carga del modelo YOLO11,
    inferencia por tiles sobre el ortomosaico, conteo por parcela (point-in-polygon)
    y cálculo de densidad. Los objetos de QGIS se convierten a GeoDataFrame
    ANTES de crear la tarea (en el hilo principal), por lo que run() solo
    trabaja con tipos de GeoPandas/Shapely, seguros entre hilos.
    """

    def __init__(self, params: dict):
        super().__init__("Detección de plantas (YOLO11)", QgsTask.Flag.CanCancel)
        self.params = params
        self.result = None
        self.error = None

    def run(self) -> bool:
        try:
            self.result = self._run_detection()
            return True
        except Exception as e:
            logger.error(f"Fallo en la tarea de detección: {e}")
            self.error = e
            return False

    def _run_detection(self) -> dict:
        p = self.params

        detector = PlantDetector(
            weights_path=p["weights_path"],
            device=p["device"],
            conf_threshold=p["conf_threshold"],
            iou_threshold=p["iou_threshold"],
            imgsz=p.get("imgsz", 640)
        )
        detector.load_model()

        with RasterManager() as raster_manager:
            raster_manager.open(p["raster_path"])
            raster_crs = raster_manager.dataset.crs.to_string()

            # Limitar la inferencia a la extensión de las parcelas: sin esto, un
            # ortomosaico de varios GB se procesaría por completo aunque solo
            # interese una fracción pequeña de su área (ej. unas pocas parcelas).
            parcelas_bounds_raster_crs = p["parcelas_gdf"].to_crs(raster_crs).total_bounds

            detections = detector.predict_orthomosaic(
                raster_manager,
                tile_size=p["tile_size"],
                overlap=p["overlap"],
                progress_callback=self._on_tile_progress,
                bounds=tuple(parcelas_bounds_raster_crs),
                should_stop=self.isCanceled
            )

        plantas_gdf = detections_to_geodataframe(detections, crs=raster_crs)

        parcelas_gdf: gpd.GeoDataFrame = p["parcelas_gdf"].copy()
        parcelas_gdf["geometry"] = parcelas_gdf["geometry"].apply(
            lambda g: g if GeometryValidator.is_valid(g) else GeometryValidator.repair(g)
        )

        id_col = "name" if "name" in parcelas_gdf.columns else parcelas_gdf.columns[0]

        # SpatialAnalysis.point_in_polygon exige el mismo CRS en ambas capas
        if plantas_gdf.crs != parcelas_gdf.crs and len(plantas_gdf) > 0:
            plantas_gdf = plantas_gdf.to_crs(parcelas_gdf.crs)

        counts = SpatialAnalysis.point_in_polygon(plantas_gdf, parcelas_gdf, polygon_id_col=id_col)

        crs_str = parcelas_gdf.crs.to_string()
        parcelas_gdf["area_m2"] = parcelas_gdf.geometry.apply(
            lambda g: GeometryManager.calculate_area(g, crs_str)
        )
        parcelas_gdf["num_plantas"] = parcelas_gdf[id_col].astype(str).map(counts).fillna(0).astype(int)
        parcelas_gdf["densidad_m2"] = parcelas_gdf["num_plantas"] / parcelas_gdf["area_m2"].replace(0, float("nan"))
        parcelas_gdf["densidad_clase"] = self._classify_density(parcelas_gdf["densidad_m2"])

        return {
            "plantas_gdf": plantas_gdf,
            "parcelas_gdf": parcelas_gdf,
            "counts": counts,
            "id_col": id_col
        }

    def _on_tile_progress(self, done: int, total: int) -> None:
        if total > 0:
            # Reservamos el último 10% del progreso para el conteo/densidad posteriores
            self.setProgress(min(90.0, (done / total) * 90.0))

    @staticmethod
    def _classify_density(densidad: "pd.Series"):
        ranking = densidad.rank(method="first", ascending=False, na_option="bottom")
        n = len(densidad)
        labels = []
        for value, rank in zip(densidad, ranking):
            if pd.isna(value):
                labels.append("Sin datos")
                continue
            percentile = rank / n
            if percentile <= 1 / 3:
                labels.append("Alta")
            elif percentile <= 2 / 3:
                labels.append("Media")
            else:
                labels.append("Baja")
        return labels


class VegetationIndexTask(QgsTask):
    """
    Calcula estadísticas zonales de un índice de vegetación (NDVI/NDRE) ya
    generado (ej. por el software de fotogrametría) para cada parcela, fuera
    del hilo principal de QGIS.
    """

    def __init__(self, params: dict):
        super().__init__("Cálculo de índice de vegetación", QgsTask.Flag.CanCancel)
        self.params = params
        self.result = None
        self.error = None

    def run(self) -> bool:
        try:
            self.result = self._run_zonal_stats()
            return True
        except Exception as e:
            logger.error(f"Fallo en la tarea de índice de vegetación: {e}")
            self.error = e
            return False

    def _run_zonal_stats(self) -> dict:
        p = self.params
        parcelas_result = VegetationIndexCalculator.zonal_statistics(
            raster_path=p["raster_path"],
            parcelas_gdf=p["parcelas_gdf"],
            thresholds=p.get("thresholds"),
            stat_column_prefix=p["prefix"]
        )
        return {"parcelas_gdf": parcelas_result, "prefix": p["prefix"]}
