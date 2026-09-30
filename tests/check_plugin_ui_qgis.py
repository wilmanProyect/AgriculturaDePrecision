"""Run with QGIS python: area accuracy, validation and dialog smoke checks."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qgis.core import QgsApplication, QgsVectorLayer, QgsFeature, QgsGeometry, QgsProject, QgsCoordinateTransform, QgsCoordinateReferenceSystem
from qgis.PyQt.QtWidgets import QTabWidget, QMessageBox
app = QgsApplication([], False)
app.initQgis()
from qgis.PyQt.QtGui import QFontDatabase, QFont
QFontDatabase.addApplicationFont('C:/Windows/Fonts/arial.ttf')
app.setFont(QFont('Arial', 10))
from plugin_qgis.area import measure_polygon_areas

layer = QgsVectorLayer('Polygon?crs=EPSG:32720', 'Prueba', 'memory')
feature = QgsFeature()
feature.setGeometry(QgsGeometry.fromWkt('POLYGON((432000 8197000,432100 8197000,432100 8197100,432000 8197100,432000 8197000),(432010 8197010,432010 8197020,432020 8197020,432020 8197010,432010 8197010))'))
layer.dataProvider().addFeatures([feature])
rows = measure_polygon_areas(layer)
assert abs(rows[0][1] - 9900) < 10
assert rows[0][2] == rows[0][1] / 10000
geographic = QgsVectorLayer('Polygon?crs=EPSG:4326', 'Grados', 'memory')
f = next(layer.getFeatures())
g = f.geometry()
g.transform(QgsCoordinateTransform(layer.crs(), geographic.crs(), QgsProject.instance()))
f.setGeometry(g)
geographic.dataProvider().addFeatures([f])
assert abs(measure_polygon_areas(geographic)[0][1] - rows[0][1]) < .01
try:
    measure_polygon_areas(layer, True)
    raise AssertionError('Empty selection accepted')
except ValueError:
    pass
layer.selectByIds([rows[0][0]])
assert len(measure_polygon_areas(layer, True)) == 1
bad = QgsVectorLayer('Polygon?crs=EPSG:32720', 'Invalid', 'memory')
f.setGeometry(QgsGeometry.fromWkt('POLYGON((0 0,10 10,0 10,10 0,0 0))'))
bad.dataProvider().addFeatures([f])
try:
    measure_polygon_areas(bad)
    raise AssertionError('Invalid geometry accepted')
except ValueError:
    pass
QgsProject.instance().addMapLayer(layer)
from plugin_qgis.dialog import PrecisionAgDialog
dialog = PrecisionAgDialog(None)
dialog.show()
app.processEvents()
dialog.parcels_combo.setLayer(layer)
dialog.detect_btn.click()
assert dialog.area_table.rowCount() == 1
assert dialog.area_export_btn.isEnabled()
assert dialog._task is None
assert dialog.veg_raster_combo is dialog.raster_combo
assert dialog.veg_index_combo.currentText() == 'exg'
assert dialog.veg_index_combo.findText('ndvi') == -1
for name in ('rows_btn', 'weed_detect_btn', 'paint_btn', 'weights_widget'):
    assert not getattr(dialog, name).isVisible(), name
out = Path(__file__).resolve().parents[1] / 'runs' / 'ui_refactor'
out.mkdir(parents=True, exist_ok=True)
dialog.grab().save(str(out / 'areas.png'))
dialog.findChild(QTabWidget).setCurrentIndex(1)
app.processEvents()
dialog.grab().save(str(out / 'analisis.png'))
dialog.close()
print('PASS: area in projected/geographic CRS, holes, selection, invalid geometry, UI action, hidden controls, default orthomosaic')
