# -*- coding: utf-8 -*-
"""
Módulo para realizar análisis espaciales avanzados (Point-in-Polygon, Buffers, Intersecciones).
"""

import geopandas as gpd
from typing import Dict

from core.logger import get_logger, log_execution_time
from core.exceptions import InvalidCRSError

logger = get_logger(__name__)

class SpatialAnalysis:
    """Clase responsable de realizar análisis espaciales ráster-vector y vector-vector."""

    @staticmethod
    def _verify_matching_crs(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> None:
        """
        Verifica que ambas capas tengan el mismo Sistema de Referencia de Coordenadas (CRS).
        """
        if gdf1.crs != gdf2.crs:
            raise InvalidCRSError(
                f"Sistemas de referencia no coinciden: '{gdf1.crs}' y '{gdf2.crs}'. "
                "Reproyecte las capas al mismo CRS antes de realizar el análisis."
            )

    @classmethod
    @log_execution_time(logger)
    def point_in_polygon(
        cls, 
        points_gdf: gpd.GeoDataFrame, 
        polygons_gdf: gpd.GeoDataFrame, 
        polygon_id_col: str = 'name'
    ) -> Dict[str, int]:
        """
        Determina qué puntos (plantas detectadas) están contenidos en qué polígonos (parcelas)
        y devuelve un conteo agrupado por el identificador de parcela especificado.
        """
        if points_gdf.empty or polygons_gdf.empty:
            logger.info("Una de las capas de entrada está vacía. Conteo finalizado en 0.")
            if not polygons_gdf.empty and polygon_id_col in polygons_gdf.columns:
                return {str(val): 0 for val in polygons_gdf[polygon_id_col]}
            return {}

        cls._verify_matching_crs(points_gdf, polygons_gdf)

        # Si la columna identificadora no existe, usamos el índice de fila de las parcelas
        col_to_use = polygon_id_col
        polygons_copy = polygons_gdf.copy()
        if col_to_use not in polygons_copy.columns:
            logger.warning(f"La columna '{polygon_id_col}' no existe en las parcelas. Usando el índice como identificador.")
            polygons_copy['temp_index_id'] = polygons_copy.index.astype(str)
            col_to_use = 'temp_index_id'

        try:
            # Realizar spatial join
            joined = gpd.sjoin(points_gdf, polygons_copy, how='inner', predicate='within')
        except Exception as e:
            logger.error(f"Fallo durante el Spatial Join: {e}")
            raise

        # Agrupar y contar
        counts = joined.groupby(col_to_use).size().to_dict()

        # Asegurar que todas las parcelas originales estén en el resultado, incluso con 0 plantas
        result = {}
        for _, row in polygons_copy.iterrows():
            key = str(row[col_to_use])
            result[key] = counts.get(row[col_to_use], 0)

        logger.info(f"Conteo Point-in-Polygon finalizado. {sum(result.values())} plantas asignadas a parcelas.")
        return result

    @classmethod
    @log_execution_time(logger)
    def calculate_buffer(cls, gdf: gpd.GeoDataFrame, distance: float) -> gpd.GeoDataFrame:
        """
        Genera un buffer alrededor de las geometrías de la capa.
        Si la capa está en CRS geográfico, advierte que la distancia del buffer será interpretada en grados,
        por lo que es altamente recomendable reproyectar a un CRS métrico antes de llamar a esta función.
        """
        if gdf.crs and gdf.crs.is_geographic:
            logger.warning(
                "La capa tiene un CRS geográfico (grados). "
                f"El buffer de distancia {distance} se creará en GRADOS y no en METROS."
            )
        
        buffered_gdf = gdf.copy()
        buffered_gdf.geometry = gdf.geometry.buffer(distance)
        return buffered_gdf

    @classmethod
    @log_execution_time(logger)
    def overlay_intersection(cls, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Realiza la intersección espacial completa de dos capas de polígonos (overlay intersection).
        Retorna una nueva capa con las áreas superpuestas y los atributos combinados.
        """
        cls._verify_matching_crs(gdf1, gdf2)
        try:
            intersection = gpd.overlay(gdf1, gdf2, how='intersection')
            return intersection
        except Exception as e:
            logger.error(f"Error al calcular overlay de intersección: {e}")
            raise
