import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
from qgis.core import QgsApplication, QgsVectorLayer, QgsFillSymbol, QgsSingleSymbolRenderer
app=QgsApplication([],False)
app.initQgis()
path=Path(__file__).resolve().parent/'maleza_similar_filtrada.gpkg'
layer=QgsVectorLayer(str(path),'Maleza similar a muestra - revisar','ogr')
assert layer.isValid()
symbol=QgsFillSymbol.createSimple({'style':'no','outline_color':'255,30,30,255','outline_width':'.35','outline_style':'solid'})
layer.setRenderer(QgsSingleSymbolRenderer(symbol))
metadata=layer.metadata()
metadata.setAbstract('Candidatos por similitud RGB con la maleza marcada por el usuario (poligono id=2). Limitados al lote id=1. No utiliza lineas ni fallas de siembra. Una muestra; requiere validacion.')
layer.setMetadata(metadata)
print(layer.saveNamedStyle(str(path.with_suffix('.qml'))))
