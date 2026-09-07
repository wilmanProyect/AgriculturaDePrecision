# -*- coding: utf-8 -*-
"""
Módulo vectorial del Motor GIS.
"""

from .vector_metadata import VectorMetadata
from .vector_manager import VectorManager
from .vector_utils import parse_kml_fallback

__all__ = [
    'VectorMetadata',
    'VectorManager',
    'parse_kml_fallback'
]
