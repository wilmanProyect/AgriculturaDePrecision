"""Run explicitly with python-qgis.bat (not regular pytest)."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plugin_qgis  # initializes user dependency paths
from qgis.core import QgsApplication, QgsProject
from qgis.gui import QgsMapCanvas
from plugin_qgis.dialog import PrecisionAgDialog
from plugin_qgis.worker import RowAnalysisTask
from core.analysis.row_detection import RowOptions
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

def scene(folder):
    yy, xx = np.indices((160, 320))
    vegetation = np.zeros((160, 320), bool)
    for row in range(20, 141, 20):
        vegetation |= np.abs(yy-row) <= 2
    vegetation[58:63, 120:160] = False
    rgb = np.full((3, 160, 320), 100, dtype='uint8')
    rgb[0, vegetation], rgb[1, vegetation], rgb[2, vegetation] = 65, 180, 55
    path = folder/'scene.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=320, height=160, count=3,
                       dtype='uint8', crs='EPSG:32720', transform=from_origin(500000, 8200000, .05, .05)) as dst:
        dst.write(rgb)
    return path, gpd.GeoDataFrame({'id': [1]}, geometry=[box(500000.1,8199992.1,500015.9,8199999.9)], crs=32720)

app = QgsApplication([], False)
app.initQgis()
canvas = QgsMapCanvas()

class Iface:
    def mapCanvas(self):
        return canvas

with tempfile.TemporaryDirectory() as folder:
    path, parcels = scene(Path(folder))
    task = RowAnalysisTask(dict(raster_path=str(path), parcelas_gdf=parcels,
                                options=RowOptions(spacing_m=1), output_dir=folder))
    assert task.run(), task.error
    assert task.result['rows'] >= 6 and task.result['gaps'] == 1, task.result
    dialog = PrecisionAgDialog(Iface())
    assert dialog.rows_btn.isEnabled()
    dialog._row_task = task
    dialog._on_rows_completed()
    layers = list(QgsProject.instance().mapLayers().values())
    assert len(layers) == 2
    assert all(layer.isValid() and layer.geometryType() == 1 for layer in layers)
    assert {layer.renderer().symbol().color().name() for layer in layers} == {'#00dce8', '#ff2525'}
    assert len(list(Path(folder).rglob('*.qml'))) == 2
    dialog.show()
    app.processEvents()
    dialog.grab().save(str(Path(__file__).resolve().parents[1]/'runs'/'plugin_surcos_ui.png'))
    dialog.close()
    QgsProject.instance().removeAllMapLayers()
    layers.clear()
    del dialog, task
print('QGIS smoke OK: task, GeoPackages, button, line layers and colors')
