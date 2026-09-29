import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
from qgis.core import QgsApplication,QgsVectorLayer,QgsFillSymbol,QgsSingleSymbolRenderer
app=QgsApplication([],False);app.initQgis()
path=Path(__file__).resolve().parent/'maleza_revision_conservadora.gpkg'
layer=QgsVectorLayer(str(path),'Ensayo maleza vs soya - revisar','ogr')
assert layer.isValid()
layer.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({
    'style':'no','outline_color':'255,30,30,255','outline_width':'.35','outline_style':'solid'})))
metadata=layer.metadata()
metadata.setAbstract('Ensayo supervisado: 9 muestras de maleza y 5 de soya. Consenso de dos clasificadores RGB/textura y filtro de forma. Limitado a poligono id=1; no usa lineas ni fallas. Sin validacion independiente; fuerte omision de maleza. Candidatos para revision, no inventario de infestacion.')
layer.setMetadata(metadata)
print(layer.saveNamedStyle(str(path.with_suffix('.qml'))))
