"""Create QGIS styles for the separate review layers."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
from qgis.core import QgsApplication,QgsVectorLayer,QgsLineSymbol,QgsSingleSymbolRenderer
app=QgsApplication([],False);app.initQgis()
out=Path(__file__).resolve().parents[2]/'runs/rows_revision_20260930'
for filename,color,width in [('lineas_revision','0,210,220,255','.20'),('fallas_revision','255,30,30,255','.4')]:
    path=out/(filename+'.gpkg')
    layer=QgsVectorLayer(str(path),filename,'ogr')
    assert layer.isValid()
    layer.setRenderer(QgsSingleSymbolRenderer(QgsLineSymbol.createSimple({'line_color':color,'line_width':width})))
    meta=layer.metadata()
    meta.setAbstract('Revision RGB 0.3.1: contexto solapado y ajuste conservador entre bloques. Celeste solo en tramos con evidencia vegetal; rojo en vacios internos. Misma resolucion 0.02659 m/pixel, separacion 0.45 m y longitud minima de falla 0.2 m del analisis anterior. Sin validacion agronomica; revisar omisiones, malezas y sombras.')
    layer.setMetadata(meta)
    message,ok=layer.saveNamedStyle(str(path.with_suffix('.qml')))
    assert ok,message
    print(filename,layer.featureCount())
