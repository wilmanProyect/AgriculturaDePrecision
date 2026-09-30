"""Local geospatial QA figure: source image, previous result and revision."""
from pathlib import Path
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from shapely.geometry import box
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]
out=root/'runs/rows_revision_20260930'
old=Path('C:/Users/VICTUS/Documents/Agroptima/AnalisisSiembra/20260929_133554_a6cd93')
datasets=[gpd.read_file(old/'lineas_siembra.gpkg'),gpd.read_file(old/'posibles_fallas.gpkg'),
          gpd.read_file(out/'lineas_revision.gpkg'),gpd.read_file(out/'fallas_revision.gpkg')]
for index,(cx,cy,dx,dy) in enumerate([(-63.63136841,-16.30712973,.000105,.000065),
                                    (-63.63015,-16.3064,.00014,.000085)]):
    bounds=(cx-dx,cy-dy,cx+dx,cy+dy)
    region=box(*bounds)
    with rasterio.open('C:/Users/VICTUS/Downloads/result.tif') as src:
        window=from_bounds(*bounds,src.transform)
        rgb=src.read([1,2,3],window=window,out_shape=(3,650,1050),resampling=Resampling.bilinear)
    fig,axes=plt.subplots(1,3,figsize=(21,7))
    for ax,title in zip(axes,['Ortomosaico','Anterior: ejes completos','Revision: vegetacion y vacios']):
        ax.imshow(rgb.transpose(1,2,0),extent=(bounds[0],bounds[2],bounds[1],bounds[3]))
        ax.set_title(title);ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.axis('off')
    for ax,pair in [(axes[1],datasets[:2]),(axes[2],datasets[2:])]:
        for frame,color,width in zip(pair,['cyan','red'],[.6,1.2]):
            local=frame.to_crs(4326)
            local=local[local.intersects(region)]
            for line in local.geometry:
                coords=np.asarray(line.coords)
                ax.plot(coords[:,0],coords[:,1],color=color,lw=width)
    fig.tight_layout();fig.savefig(out/f'comparacion_{index+1}.png',dpi=130);plt.close(fig)

rows,gaps=datasets[2:]
metrics={'tracks_crossing_tiles':int((rows.groupby('surco_id').parcela_id.nunique()>1).sum()),
         'row_segments':len(rows),'gap_segments':len(gaps)}
(out/'continuidad.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics))
