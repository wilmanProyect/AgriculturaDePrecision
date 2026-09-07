# -*- coding: utf-8 -*-
"""
Excepciones personalizadas para el Motor GIS.
"""

class PrecisionAgError(Exception):
    """Clase base para todas las excepciones del sistema."""
    pass

class RasterNotFoundError(PrecisionAgError):
    """Excepción lanzada cuando no se encuentra el archivo ráster."""
    pass

class InvalidCRSError(PrecisionAgError):
    """Excepción lanzada cuando el CRS es inválido, inexistente o incompatible."""
    pass

class InvalidGeometryError(PrecisionAgError):
    """Excepción lanzada cuando se encuentra una geometría corrupta o inválida."""
    pass

class UnsupportedFormatError(PrecisionAgError):
    """Excepción lanzada cuando el formato de archivo vectorial o ráster no está soportado."""
    pass

class ModelNotFoundError(PrecisionAgError):
    """Excepción lanzada cuando no se encuentra el archivo de pesos del modelo de IA."""
    pass

class ModelLoadError(PrecisionAgError):
    """Excepción lanzada cuando el modelo de IA no puede cargarse correctamente."""
    pass

class InferenceError(PrecisionAgError):
    """Excepción lanzada cuando falla el proceso de inferencia sobre una imagen."""
    pass

class TrainingError(PrecisionAgError):
    """Excepción lanzada cuando falla el proceso de entrenamiento de un modelo."""
    pass

class InvalidDatasetError(PrecisionAgError):
    """Excepción lanzada cuando la estructura o el contenido del dataset de entrenamiento es inválido."""
    pass
