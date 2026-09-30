"""Re-run the user's field into a separate review folder; never overwrite layers."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import geopandas as gpd
from shapely import wkt
from core.analysis.row_detection import RowDetector, RowOptions

root=Path(__file__).resolve().parents[2]
out=root/'runs'/'rows_revision_final_20260930';out.mkdir(exist_ok=True)
source=json.loads((root/'runs/maleza_20260929/referencia_usuario.json').read_text())
boundary=max([wkt.loads(i['geometry']['wkt']) for i in source['items']],key=lambda g:g.area)
parcels=gpd.GeoDataFrame({'id':[1]},geometry=[boundary],crs=32720)
options=RowOptions(resolution_m=.026590172169540186,spacing_m=.45,min_gap_m=.2)
result=RowDetector(options).analyze('C:/Users/VICTUS/Downloads/result.tif',parcels,crs=parcels.crs,
    progress=lambda v:print(f'{v:.0f}%',flush=True))
for key,filename in [('rows_gdf','lineas_revision'),('gaps_gdf','fallas_revision')]:
    frame=result[key]
    if len(frame):frame.to_file(out/(filename+'.gpkg'),driver='GPKG')
    assert frame.is_valid.all()
    assert frame.difference(boundary.buffer(1e-6)).is_empty.all()
summary={'rows':len(result['rows_gdf']),'gaps':len(result['gaps_gdf']),
         'vegetation_length_m':float(result['rows_gdf'].length.sum()),
         'gap_length_m':float(result['gaps_gdf'].length.sum()),'warnings':result['warnings'],
         'note':'Context overlap and strict centerline vegetation. Experimental RGB; no agronomic ground truth.',
         'all_inside':True,'valid_geometries':True}
(out/'resumen.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
