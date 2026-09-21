# -*- coding: utf-8 -*-
"""
Cálculo de índices de vegetación (NDVI, NDRE) y estadísticas zonales por parcela.

Soporta dos flujos de trabajo:
- Leer un ráster de índice ya calculado (ej. NDVI/NDRE exportado por el software
  de fotogrametría) y obtener estadísticas por parcela directamente.
- Calcular NDVI/NDRE a partir de las bandas crudas de un ortomosaico multiespectral
  (NIR, Red, RedEdge) cuando no se dispone de un producto de índice ya generado.
"""

import os
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats

from core.exceptions import RasterNotFoundError
from core.logger import get_logger, log_execution_time

logger = get_logger(__name__)

# Umbrales estándar de clasificación de vigor vegetal según NDVI.
# Referencia agronómica habitual: > 0.6 vegetación densa y sana, 0.3-0.6 moderada,
# 0.1-0.3 escasa/estresada, < 0.1 suelo desnudo o agua.
DEFAULT_NDVI_THRESHOLDS = {"Vigorosa": 0.6, "Moderada": 0.3, "Estresada": 0.1}

# ExG y VARI se calculan solo con bandas RGB (sin NIR/RedEdge), por lo que sirven
# de aproximación cuando aún no se dispone de un ortomosaico multiespectral. Su
# rango y sensibilidad no son tan estandarizados como el de NDVI: estos umbrales
# son un punto de partida razonable y conviene calibrarlos con observaciones de
# campo del cultivo concreto.
DEFAULT_EXG_THRESHOLDS = {"Vigorosa": 0.20, "Moderada": 0.05, "Estresada": -0.05}
DEFAULT_VARI_THRESHOLDS = {"Vigorosa": 0.20, "Moderada": 0.05, "Estresada": -0.10}

DEFAULT_THRESHOLDS_BY_INDEX = {
    "ndvi": DEFAULT_NDVI_THRESHOLDS,
    "ndre": DEFAULT_NDVI_THRESHOLDS,
    "exg": DEFAULT_EXG_THRESHOLDS,
    "vari": DEFAULT_VARI_THRESHOLDS,
}

# Índices calculables directamente desde un ortomosaico RGB normal (sin NIR/RedEdge).
RGB_ONLY_INDICES = ("exg", "vari")


class VegetationIndexCalculator:
    """Clase responsable de calcular índices de vegetación y estadísticas zonales por parcela."""

    @staticmethod
    def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calcula NDVI = (NIR - Red) / (NIR + Red) a partir de las bandas crudas."""
        nir = nir.astype(np.float32)
        red = red.astype(np.float32)
        denominator = nir + red
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(denominator != 0, (nir - red) / denominator, np.nan)
        return ndvi

    @staticmethod
    def compute_ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
        """Calcula NDRE = (NIR - RedEdge) / (NIR + RedEdge) a partir de las bandas crudas."""
        nir = nir.astype(np.float32)
        red_edge = red_edge.astype(np.float32)
        denominator = nir + red_edge
        with np.errstate(divide="ignore", invalid="ignore"):
            ndre = np.where(denominator != 0, (nir - red_edge) / denominator, np.nan)
        return ndre

    @staticmethod
    def compute_exg(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Calcula ExG (Excess Green) = 2g - r - b, sobre coordenadas cromáticas
        normalizadas (r = R/(R+G+B), etc.). Aproximación de vigor vegetal que solo
        necesita bandas RGB, útil cuando no se dispone de NIR/RedEdge.
        """
        red = red.astype(np.float32)
        green = green.astype(np.float32)
        blue = blue.astype(np.float32)
        total = red + green + blue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(total != 0, red / total, np.nan)
            g = np.where(total != 0, green / total, np.nan)
            b = np.where(total != 0, blue / total, np.nan)
        return 2 * g - r - b

    @staticmethod
    def compute_vari(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Calcula VARI (Visible Atmospherically Resistant Index) = (G - R) / (G + R - B).
        Aproximación de vigor vegetal que solo necesita bandas RGB.
        """
        red = red.astype(np.float32)
        green = green.astype(np.float32)
        blue = blue.astype(np.float32)
        denominator = green + red - blue
        with np.errstate(divide="ignore", invalid="ignore"):
            vari = np.where(denominator != 0, (green - red) / denominator, np.nan)
        return vari

    @staticmethod
    def classify_value(value: Optional[float], thresholds: Dict[str, float] = None) -> str:
        """
        Clasifica un valor de índice (ej. NDVI medio de una parcela) según umbrales
        ordenados de mayor a menor. La última categoría implícita es "Suelo/Agua".
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "Sin datos"

        thresholds = thresholds or DEFAULT_NDVI_THRESHOLDS
        for label, threshold in sorted(thresholds.items(), key=lambda item: -item[1]):
            if value >= threshold:
                return label
        return "Suelo/Agua"

    @classmethod
    @log_execution_time(logger)
    def zonal_statistics(
        cls,
        raster_path: str,
        parcelas_gdf: gpd.GeoDataFrame,
        band: int = 1,
        thresholds: Dict[str, float] = None,
        stat_column_prefix: str = "ndvi"
    ) -> gpd.GeoDataFrame:
        """
        Calcula estadísticas zonales (media, mínimo, máximo, desviación estándar) del
        índice de vegetación dentro de cada parcela, y clasifica cada una según el
        valor medio. Reproyecta las parcelas al CRS del ráster si es necesario.

        Devuelve una copia del GeoDataFrame de parcelas con las columnas:
        `{prefix}_mean`, `{prefix}_min`, `{prefix}_max`, `{prefix}_std`, `{prefix}_clase`.
        """
        if not os.path.exists(raster_path):
            raise RasterNotFoundError(f"El ráster de índice de vegetación no existe: {raster_path}")

        import rasterio
        with rasterio.open(raster_path) as dataset:
            raster_crs = dataset.crs.to_string()

        parcelas = parcelas_gdf.copy()
        if parcelas.crs is not None and parcelas.crs.to_string() != raster_crs:
            parcelas = parcelas.to_crs(raster_crs)

        stats = zonal_stats(
            parcelas,
            raster_path,
            band=band,
            stats=["mean", "min", "max", "std"],
            nodata=np.nan,
            geojson_out=False
        )

        parcelas = cls._assign_stats(parcelas, stats, thresholds, stat_column_prefix)

        logger.info(
            f"Estadísticas zonales de '{stat_column_prefix}' calculadas para {len(parcelas)} parcela(s)."
        )
        return parcelas.to_crs(parcelas_gdf.crs) if parcelas_gdf.crs is not None else parcelas

    @classmethod
    @log_execution_time(logger)
    def zonal_statistics_rgb_index(
        cls,
        raster_path: str,
        parcelas_gdf: gpd.GeoDataFrame,
        index: str = "exg",
        red_band: int = 1,
        green_band: int = 2,
        blue_band: int = 3,
        thresholds: Dict[str, float] = None,
        stat_column_prefix: str = None
    ) -> gpd.GeoDataFrame:
        """
        Calcula un índice de vegetación aproximado (ExG o VARI) directamente a partir
        de las bandas RGB de un ortomosaico normal, sin necesitar NIR/RedEdge, y
        obtiene sus estadísticas zonales por parcela igual que `zonal_statistics`.
        """
        index = index.lower()
        if index not in RGB_ONLY_INDICES:
            raise ValueError(f"Índice RGB no soportado: '{index}'. Usa uno de {RGB_ONLY_INDICES}.")

        if not os.path.exists(raster_path):
            raise RasterNotFoundError(f"El ortomosaico no existe: {raster_path}")

        stat_column_prefix = stat_column_prefix or index
        thresholds = thresholds or DEFAULT_THRESHOLDS_BY_INDEX.get(index)

        import rasterio

        with rasterio.open(raster_path) as dataset:
            raster_crs = dataset.crs.to_string()
            transform = dataset.transform
            nodata = dataset.nodata
            red = dataset.read(red_band)
            green = dataset.read(green_band)
            blue = dataset.read(blue_band)

        if index == "exg":
            index_array = cls.compute_exg(red, green, blue)
        else:
            index_array = cls.compute_vari(red, green, blue)

        if nodata is not None:
            invalid = (red == nodata) & (green == nodata) & (blue == nodata)
            index_array = np.where(invalid, np.nan, index_array)

        parcelas = parcelas_gdf.copy()
        if parcelas.crs is not None and parcelas.crs.to_string() != raster_crs:
            parcelas = parcelas.to_crs(raster_crs)

        stats = zonal_stats(
            parcelas,
            index_array,
            affine=transform,
            stats=["mean", "min", "max", "std"],
            nodata=np.nan,
            geojson_out=False
        )

        parcelas = cls._assign_stats(parcelas, stats, thresholds, stat_column_prefix)

        logger.info(
            f"Estadísticas zonales de '{stat_column_prefix}' (RGB) calculadas para {len(parcelas)} parcela(s)."
        )
        return parcelas.to_crs(parcelas_gdf.crs) if parcelas_gdf.crs is not None else parcelas

    @classmethod
    def _assign_stats(
        cls,
        parcelas: gpd.GeoDataFrame,
        stats: list,
        thresholds: Dict[str, float],
        stat_column_prefix: str
    ) -> gpd.GeoDataFrame:
        """Vuelca una lista de estadísticas zonales (de rasterstats) como columnas del GeoDataFrame."""
        parcelas[f"{stat_column_prefix}_mean"] = [s["mean"] for s in stats]
        parcelas[f"{stat_column_prefix}_min"] = [s["min"] for s in stats]
        parcelas[f"{stat_column_prefix}_max"] = [s["max"] for s in stats]
        parcelas[f"{stat_column_prefix}_std"] = [s["std"] for s in stats]
        parcelas[f"{stat_column_prefix}_clase"] = [
            cls.classify_value(s["mean"], thresholds) for s in stats
        ]
        return parcelas
