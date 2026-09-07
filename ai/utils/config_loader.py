# -*- coding: utf-8 -*-
"""
Carga de la configuración del Motor IA (config/default.yaml, sección 'ai').
"""

import os
from typing import Any, Dict

import yaml

from core.logger import get_logger

logger = get_logger(__name__)


def load_ai_config(config_path: str) -> Dict[str, Any]:
    """
    Lee el archivo YAML de configuración y devuelve la sección 'ai'.
    Si el archivo o la sección no existen, devuelve un diccionario vacío.
    """
    if not os.path.exists(config_path):
        logger.warning(f"Archivo de configuración no encontrado: {config_path}. Se usarán valores por defecto.")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f) or {}

    return full_config.get("ai", {})
