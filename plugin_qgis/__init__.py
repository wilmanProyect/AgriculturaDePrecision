# -*- coding: utf-8 -*-
"""
Punto de entrada del plugin de QGIS para Agricultura de Precisión.
QGIS llama a classFactory() al activar el plugin desde la pestaña Complementos.
"""

import os
import site
import sys


def _ensure_dependencies_on_path() -> None:
    """
    Al lanzar QGIS desde su acceso directo normal (qgis-bin.exe), el intérprete
    de Python embebido fija PYTHONHOME explícitamente y NO añade automáticamente
    el directorio de paquetes de usuario (a diferencia de python-qgis.bat, usado
    durante el desarrollo). Como las dependencias del plugin (torch, ultralytics,
    rasterio, fiona, y el propio paquete core/ai del proyecto) se instalaron ahí
    con `pip install --user`, las añadimos aquí explícitamente para que el plugin
    funcione sin importar cómo se haya iniciado QGIS.
    """
    candidates = []

    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass

    appdata = os.environ.get("APPDATA")
    if appdata:
        py_tag = f"Python{sys.version_info.major}{sys.version_info.minor}"
        candidates.append(os.path.join(appdata, "Python", py_tag, "site-packages"))

    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_ensure_dependencies_on_path()


def classFactory(iface):
    from .plugin import PrecisionAgPlugin
    return PrecisionAgPlugin(iface)
