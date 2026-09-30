# Interfaz simplificada (0.3.0)

Selecciona la capa de polígonos y elige una pestaña:

- **Área de polígonos:** mide todos los elementos o solo los seleccionados en el mapa. Muestra m², hectáreas y suma total; permite exportar la medición a CSV. No requiere ortomosaico ni pesos de IA. Lee las geometrías actuales sin modificar la capa original.
- **Análisis del ortomosaico:** conserva líneas y posibles fallas RGB e índices ExG/VARI. Ambas herramientas utilizan el mismo ortomosaico. Los resultados de líneas siguen siendo estimaciones que requieren revisión.

La medición utiliza el elipsoide WGS84, transforma desde el CRS de origen y respeta huecos y polígonos multipartes. Rechaza geometrías vacías, inválidas o sin CRS. La suma no disuelve solapes. El CSV refleja la última medición: vuelve a calcular después de editar geometrías o cambiar la selección.

Se ocultan el conteo de plantas, densidad, detección de malezas, opciones de modelos YOLO y el selector separado de ráster de índice. NDVI/NDRE no se ofrecen en este flujo RGB. El código de esas operaciones se conserva para trabajo posterior.

## Verificación

Ejecutar `tests/check_plugin_ui_qgis.py` con el intérprete Python de QGIS. Comprueba medición en CRS geográfico/proyectado, huecos, selección, geometrías inválidas, el botón de área y controles ocultos. Genera capturas de ambas pestañas en `runs/ui_refactor/`.

El ZIP instala el plugin; los análisis RGB mantienen las dependencias del proyecto core/ai del entorno existente. La actualización requiere reinstalar el ZIP y reiniciar QGIS para cargar el nuevo código.
