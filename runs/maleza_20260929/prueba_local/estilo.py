import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
from qgis.core import QgsApplication,QgsVectorLayer,QgsFillSymbol,QgsSingleSymbolRenderer
app=QgsApplication([],False);app.initQgis()
path=Path(__file__).resolve().parent/'candidatos_prueba.gpkg'
layer=QgsVectorLayer(str(path),'Prueba local - sin validar','ogr')
assert layer.isValid()
layer.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({'style':'no','outline_color':'255,30,30,255','outline_width':'.35','outline_style':'solid'})))
meta=layer.metadata();meta.setAbstract('Ensayo del clasificador V4 RGB y textura en Poligono de prueba. Umbral 0.98, area minima 0.06 m2. No hay muestras previas de maleza dentro del limite. Resultado sin validar; no usar como mapa de infestacion. No utiliza lineas ni fallas de siembra.');layer.setMetadata(meta)
print(layer.saveNamedStyle(str(path.with_suffix('.qml'))))
