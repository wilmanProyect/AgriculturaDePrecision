# -*- coding: utf-8 -*-
"""
Tarea en segundo plano (QgsTask) que ejecuta la detección YOLO11 y el análisis
espacial sin bloquear la interfaz de QGIS.
"""

import geopandas as gpd
import pandas as pd

from qgis.core import QgsTask

from ai.inference.detector import PlantDetector
from ai.utils.georeferencing import detections_to_geodataframe, detections_to_polygon_geodataframe
from core.analysis.spatial_analysis import SpatialAnalysis
from core.analysis.vegetation_index import RGB_ONLY_INDICES, VegetationIndexCalculator
from core.geometry.geometry_manager import GeometryManager
from core.geometry.geometry_validator import GeometryValidator
from core.gis.raster.raster_manager import RasterManager
from core.logger import get_logger

logger = get_logger(__name__)


class RowAnalysisTask(QgsTask):
    """Analiza surcos fuera del hilo de QGIS y persiste sus resultados."""

    def __init__(self, params: dict):
        super().__init__("Detectar líneas de siembra y fallas", QgsTask.Flag.CanCancel)
        self.params = params
        self.result = None
        self.error = None

    def run(self) -> bool:
        from core.analysis.row_detection import RowDetector, RowAnalysisCanceled
        try:
            import json
            from dataclasses import asdict
            from datetime import datetime
            from pathlib import Path
            from uuid import uuid4

            # `crs`, si se pasó, ya fue resuelto en el hilo principal (ver dialog.py):
            # calcular GeoDataFrame.estimate_utm_crs()/to_crs() por primera vez desde
            # este hilo en segundo plano puede colgar el proceso con un crash nativo
            # de PROJ (access violation) en algunos entornos de QGIS.
            crs = self.params.get('crs')
            weights_path = self.params.get('weights_path')
            if weights_path:
                from ai.inference.row_segmenter import RowModelOptions, RowSegmenter
                segmenter = RowSegmenter(RowModelOptions(
                    weights_path=weights_path,
                    conf_threshold=self.params.get('conf_threshold', 0.25)
                ))
                segmenter.load_model()
                result = segmenter.analyze(
                    self.params['raster_path'], self.params['parcelas_gdf'], self.params['options'],
                    progress=lambda value: self.setProgress(value * .9), should_stop=self.isCanceled, crs=crs)
            else:
                result = RowDetector(self.params['options']).analyze(
                    self.params['raster_path'], self.params['parcelas_gdf'],
                    progress=lambda value: self.setProgress(value * .9), should_stop=self.isCanceled, crs=crs)
            if self.isCanceled():
                return False
            folder = Path(self.params['output_dir']) / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
            folder.mkdir(parents=True, exist_ok=False)
            paths = {}
            for key, filename in [('rows_gdf', 'lineas_siembra'), ('gaps_gdf', 'posibles_fallas')]:
                path = folder / (filename + '.gpkg')
                # Empty results are reported explicitly; never create a misleading point layer.
                if not result[key].empty:
                    result[key].to_file(path, layer=filename, driver='GPKG')
                    paths[key] = str(path)
            note = ('Segmentado por el modelo YOLO indicado; revisa su desempeño real antes de '
                    'usar los metros como cifra definitiva.' if weights_path else
                    'Baja vegetación RGB; no confirma falla ni identifica surcos completamente ausentes.')
            summary = {
                'method': result['method'], 'options': asdict(result['options']),
                'model_weights': weights_path or None,
                'raster': self.params['raster_path'], 'crs': str(result['rows_gdf'].crs),
                'rows': len(result['rows_gdf']), 'gaps': len(result['gaps_gdf']),
                'row_length_m': float(result['rows_gdf'].length.sum()),
                'gap_length_m': float(result['gaps_gdf'].length.sum()),
                'warnings': result['warnings'],
                'note': note}
            (folder / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
            self.result = {'paths': paths, 'folder': str(folder), **summary}
            self.setProgress(100)
            return not self.isCanceled()
        except RowAnalysisCanceled:
            return False
        except Exception as exc:
            logger.exception("Falló el análisis de surcos")
            self.error = exc
            return False


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


class WeedAnalysisTask(QgsTask):
    """
    Detecta malezas (y cultivo, si el modelo lo distingue) sobre un ortomosaico con
    un modelo YOLO entrenado para cultivo/maleza, fuera del hilo principal de QGIS.
    Reutiliza PlantDetector.predict_orthomosaic (mismo tiling georreferenciado que
    DetectionTask); lo único específico de esta tarea es cómo se clasifican y
    resumen las detecciones por parcela.
    """

    # Nombres de clase (en minúsculas) que el modelo puede usar para la maleza / el cultivo.
    WEED_LABELS = {"weed", "weeds", "maleza", "malezas"}
    CROP_LABELS = {"crop", "cultivo"}

    def __init__(self, params: dict):
        super().__init__("Detección de malezas (YOLO)", QgsTask.Flag.CanCancel)
        self.params = params
        self.result = None
        self.error = None

    def run(self) -> bool:
        try:
            self.result = self._run_weed_detection()
            return True
        except Exception as e:
            logger.error(f"Fallo en la tarea de detección de malezas: {e}")
            self.error = e
            return False

    def _run_weed_detection(self) -> dict:
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
            parcelas_bounds_raster_crs = p["parcelas_gdf"].to_crs(raster_crs).total_bounds

            detections = detector.predict_orthomosaic(
                raster_manager,
                tile_size=p["tile_size"],
                overlap=p["overlap"],
                progress_callback=self._on_tile_progress,
                bounds=tuple(parcelas_bounds_raster_crs),
                should_stop=self.isCanceled
            )

        detecciones_gdf = detections_to_geodataframe(detections, crs=raster_crs)
        # Recuadros (bounding box) de cada detección, para dibujar un polígono alrededor
        # de cada maleza en el mapa en lugar de un único punto central.
        poligonos_gdf = detections_to_polygon_geodataframe(detections, crs=raster_crs)

        parcelas_gdf: gpd.GeoDataFrame = p["parcelas_gdf"].copy()
        parcelas_gdf["geometry"] = parcelas_gdf["geometry"].apply(
            lambda g: g if GeometryValidator.is_valid(g) else GeometryValidator.repair(g)
        )
        id_col = "name" if "name" in parcelas_gdf.columns else parcelas_gdf.columns[0]

        if detecciones_gdf.crs != parcelas_gdf.crs and len(detecciones_gdf) > 0:
            detecciones_gdf = detecciones_gdf.to_crs(parcelas_gdf.crs)
            poligonos_gdf = poligonos_gdf.to_crs(parcelas_gdf.crs)

        # El conteo por parcela usa el punto central (evita descartar detecciones cuyo
        # recuadro cruza el borde de la parcela); los polígonos, construidos en el mismo
        # orden a partir de la misma lista de detecciones, se reparten con la misma máscara.
        is_weed, has_crop_class = self._split_weed_crop(detecciones_gdf)
        malezas_gdf = detecciones_gdf[is_weed].copy()
        cultivo_gdf = detecciones_gdf[~is_weed].copy()
        malezas_poly_gdf = poligonos_gdf[is_weed].copy()
        cultivo_poly_gdf = poligonos_gdf[~is_weed].copy()

        malezas_counts = SpatialAnalysis.point_in_polygon(malezas_gdf, parcelas_gdf, polygon_id_col=id_col)
        cultivo_counts = SpatialAnalysis.point_in_polygon(cultivo_gdf, parcelas_gdf, polygon_id_col=id_col)

        crs_str = parcelas_gdf.crs.to_string()
        parcelas_gdf["area_m2"] = parcelas_gdf.geometry.apply(
            lambda g: GeometryManager.calculate_area(g, crs_str)
        )
        parcelas_gdf["num_malezas"] = parcelas_gdf[id_col].astype(str).map(malezas_counts).fillna(0).astype(int)
        parcelas_gdf["num_cultivo"] = parcelas_gdf[id_col].astype(str).map(cultivo_counts).fillna(0).astype(int)

        if has_crop_class:
            # El modelo distingue cultivo de maleza: la cobertura relativa (malezas /
            # total de detecciones) es más informativa que una densidad absoluta, porque
            # no depende de cuán tupido esté el cultivo de soja en cada parcela.
            total = parcelas_gdf["num_malezas"] + parcelas_gdf["num_cultivo"]
            parcelas_gdf["cobertura_malezas"] = parcelas_gdf["num_malezas"] / total.replace(0, float("nan"))
            parcelas_gdf["infestacion_clase"] = self._classify_infestation_ratio(parcelas_gdf["cobertura_malezas"])
        else:
            # Modelo de una sola clase (solo detecta maleza): no hay forma de calcular una
            # cobertura relativa, así que se compara la densidad de malezas entre parcelas.
            parcelas_gdf["cobertura_malezas"] = (
                parcelas_gdf["num_malezas"] / parcelas_gdf["area_m2"].replace(0, float("nan"))
            )
            parcelas_gdf["infestacion_clase"] = self._classify_infestation_density(parcelas_gdf["cobertura_malezas"])

        return {
            "malezas_gdf": malezas_gdf,
            "cultivo_gdf": cultivo_gdf,
            "malezas_poly_gdf": malezas_poly_gdf,
            "cultivo_poly_gdf": cultivo_poly_gdf,
            "parcelas_gdf": parcelas_gdf,
            "id_col": id_col,
            "has_crop_class": has_crop_class,
            "counts": malezas_counts,
        }

    def _on_tile_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.setProgress(min(90.0, (done / total) * 90.0))

    @classmethod
    def _split_weed_crop(cls, detecciones_gdf: "gpd.GeoDataFrame"):
        """
        Separa las detecciones en maleza/cultivo según el nombre de clase del modelo.
        Si el modelo distingue explícitamente 'weed'/'crop' (ej. dataset crop_weed),
        se usa esa distinción. Si el modelo tiene una sola clase (ej. un detector de
        malezas dedicado, sin clase de cultivo), todas las detecciones se consideran
        malezas. Devuelve (máscara_es_maleza, hay_clase_de_cultivo).
        """
        if detecciones_gdf.empty:
            import pandas as pd
            return pd.Series([], dtype=bool), False

        names_lower = detecciones_gdf["class_name"].astype(str).str.lower()
        weed_mask = names_lower.isin(cls.WEED_LABELS)
        if weed_mask.any():
            return weed_mask, bool((~weed_mask).any())

        crop_mask = names_lower.isin(cls.CROP_LABELS)
        return ~crop_mask, bool(crop_mask.any() and (~crop_mask).any())

    @staticmethod
    def _classify_infestation_ratio(cobertura: "pd.Series"):
        """Umbrales agronómicos habituales para % de cobertura de malezas."""
        labels = []
        for value in cobertura:
            if pd.isna(value):
                labels.append("Sin datos")
            elif value < 0.10:
                labels.append("Baja")
            elif value < 0.30:
                labels.append("Media")
            else:
                labels.append("Alta")
        return labels

    @staticmethod
    def _classify_infestation_density(densidad: "pd.Series"):
        ranking = densidad.rank(method="first", ascending=True, na_option="bottom")
        n = len(densidad)
        labels = []
        for value, rank in zip(densidad, ranking):
            if pd.isna(value):
                labels.append("Sin datos")
                continue
            percentile = rank / n
            if percentile <= 1 / 3:
                labels.append("Baja")
            elif percentile <= 2 / 3:
                labels.append("Media")
            else:
                labels.append("Alta")
        return labels


class VegetationIndexTask(QgsTask):
    """
    Calcula estadísticas zonales de un índice de vegetación para cada parcela,
    fuera del hilo principal de QGIS. Soporta dos flujos:
    - NDVI/NDRE: requieren un ráster de índice ya generado (ej. por el software
      de fotogrametría a partir de un vuelo con sensor NIR/RedEdge).
    - ExG/VARI: se calculan al vuelo a partir de las bandas RGB del propio
      ortomosaico, sin necesitar NIR/RedEdge.
    """

    def __init__(self, params: dict):
        super().__init__("Cálculo de índice de vegetación", QgsTask.Flag.CanCancel)
        self.params = params
        self.result = None
        self.error = None

    def run(self) -> bool:
        try:
            if self.isCanceled():
                return False
            self.result = self._run_zonal_stats()
            self.setProgress(100)
            return not self.isCanceled()
        except Exception as e:
            logger.error(f"Fallo en la tarea de índice de vegetación: {e}")
            self.error = e
            return False

    def _run_zonal_stats(self) -> dict:
        p = self.params
        prefix = p["prefix"]

        if prefix in RGB_ONLY_INDICES:
            parcelas_result = VegetationIndexCalculator.zonal_statistics_rgb_index(
                raster_path=p["raster_path"],
                parcelas_gdf=p["parcelas_gdf"],
                index=prefix,
                thresholds=p.get("thresholds"),
                stat_column_prefix=prefix
            )
        else:
            parcelas_result = VegetationIndexCalculator.zonal_statistics(
                raster_path=p["raster_path"],
                parcelas_gdf=p["parcelas_gdf"],
                thresholds=p.get("thresholds"),
                stat_column_prefix=prefix
            )
        return {"parcelas_gdf": parcelas_result, "prefix": prefix}
