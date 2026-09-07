# -*- coding: utf-8 -*-
"""
Logger central estructurado para el proyecto.
Incluye un decorador para registrar tiempos de procesamiento.
"""

import time
import logging
from functools import wraps

# Configuración básica del logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger configurado con el nombre dado.
    """
    return logging.getLogger(name)

def log_execution_time(logger: logging.Logger = None):
    """
    Decorador para registrar el tiempo que tarda una función en ejecutarse.
    """
    def decorator(func):
        # Resolver logger por defecto si no se pasa
        log_target = logger if logger is not None else get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            log_target.info(f"Iniciando ejecución de '{func.__name__}'...")
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                log_target.info(f"Función '{func.__name__}' completada con éxito en {elapsed:.4f} segundos.")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                log_target.error(f"Función '{func.__name__}' falló tras {elapsed:.4f} segundos con error: {e}")
                raise
        return wrapper
    return decorator
