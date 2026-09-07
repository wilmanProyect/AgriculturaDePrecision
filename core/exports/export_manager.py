# -*- coding: utf-8 -*-
"""
Módulo para exportar capas geográficas y datos tabulares a múltiples formatos.
"""

import os
import pandas as pd
import geopandas as gpd
from typing import Dict, Any

from core.logger import get_logger, log_execution_time
from core.exceptions import UnsupportedFormatError

logger = get_logger(__name__)

class ExportManager:
    """Clase responsable de guardar y exportar capas de datos e informes a disco."""

    @staticmethod
    @log_execution_time(logger)
    def export_geodataframe(
        gdf: gpd.GeoDataFrame, 
        output_path: str, 
        driver: str = None
    ) -> None:
        """
        Exporta un GeoDataFrame a un formato espacial (Shapefile, GeoPackage, GeoJSON).
        El formato se autodetecta por la extensión si no se especifica el driver.
        """
        if gdf.empty:
            logger.warning("Intentando exportar un GeoDataFrame vacío.")

        ext = os.path.splitext(output_path)[1].lower()
        
        # Mapeo de extensiones a drivers de GDAL/Fiona
        drivers = {
            '.shp': 'ESRI Shapefile',
            '.gpkg': 'GPKG',
            '.geojson': 'GeoJSON',
            '.json': 'GeoJSON'
        }

        selected_driver = driver or drivers.get(ext)
        if not selected_driver:
            raise UnsupportedFormatError(
                f"Extensión de archivo '{ext}' no soportada para exportar datos vectoriales. "
                f"Extensiones soportadas: {list(drivers.keys())}"
            )

        # Crear directorios de destino si no existen
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            logger.info(f"Exportando GeoDataFrame a '{output_path}' usando driver '{selected_driver}'...")
            gdf.to_file(output_path, driver=selected_driver)
            logger.info("Exportación espacial completada con éxito.")
        except Exception as e:
            logger.error(f"Error al guardar el archivo vectorial: {e}")
            raise

    @staticmethod
    @log_execution_time(logger)
    def export_tabular_data(
        data: Dict[str, Any], 
        output_path: str,
        columns: list = None
    ) -> None:
        """
        Exporta datos de tipo clave-valor (como resultados de conteo por parcela)
        a formatos tabulares como CSV o Excel (.xlsx).
        """
        ext = os.path.splitext(output_path)[1].lower()
        if ext not in ['.csv', '.xlsx']:
            raise UnsupportedFormatError(
                f"Formato tabular '{ext}' no soportado. Formatos válidos: ['.csv', '.xlsx']"
            )

        # Convertir diccionario a DataFrame de Pandas
        cols = columns or ['Clave', 'Valor']
        df = pd.DataFrame(list(data.items()), columns=cols)

        # Crear directorios si no existen
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            if ext == '.csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif ext == '.xlsx':
                df.to_excel(output_path, index=False)
            logger.info(f"Datos tabulares exportados con éxito a '{output_path}'.")
        except Exception as e:
            logger.error(f"Error al guardar datos tabulares: {e}")
            raise
