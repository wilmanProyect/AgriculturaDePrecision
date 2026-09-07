# -*- coding: utf-8 -*-
"""
Pruebas unitarias para las excepciones personalizadas del sistema.
"""

import pytest
from core.exceptions import (
    PrecisionAgError,
    RasterNotFoundError,
    InvalidCRSError,
    InvalidGeometryError,
    UnsupportedFormatError
)

def test_exceptions_inheritance():
    """Verifica que todas las excepciones heredan de la clase base y de Exception."""
    assert issubclass(RasterNotFoundError, PrecisionAgError)
    assert issubclass(InvalidCRSError, PrecisionAgError)
    assert issubclass(InvalidGeometryError, PrecisionAgError)
    assert issubclass(UnsupportedFormatError, PrecisionAgError)
    assert issubclass(PrecisionAgError, Exception)

def test_raise_exceptions():
    """Verifica que se pueden lanzar y capturar las excepciones correctamente."""
    with pytest.raises(RasterNotFoundError):
        raise RasterNotFoundError("Ráster no encontrado")
        
    with pytest.raises(InvalidCRSError):
        raise InvalidCRSError("CRS no válido")
        
    with pytest.raises(InvalidGeometryError):
        raise InvalidGeometryError("Geometría corrupta")
        
    with pytest.raises(UnsupportedFormatError):
        raise UnsupportedFormatError("Formato no soportado")
