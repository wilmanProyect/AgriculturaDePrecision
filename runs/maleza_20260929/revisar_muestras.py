from pathlib import Path
import json
import numpy as np
import geopandas as gpd
from shapely import wkt
from affine import Affine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask

out=Path(__file__).resolve().parent
j=json.loads((out/'muestras_nuevas.json').read_text())
g=gpd.GeoDataFrame({'id':[i['attributes']['id'] for i in j['items']]},geometry=[wkt.loads(i['geometry']['wkt']) for i in j['items']],crs=4326).to_crs(32720)
g.geometry=g.geometry.make_valid()
g=g.sort_values('id')
g.to_file(out/'muestras_maleza_usuario_v2.gpkg',layer='maleza_confirmada_usuario',driver='GPKG')
z=np.load(out/'crop.npz'); data=z['data']; tr=Affine(*z['tr']); inv=~tr
fig,axes=plt.subplots(3,3,figsize=(15,15))
stats=[]
for ax,(_,row) in zip(axes.flat,g.iterrows()):
    x,y=inv*(row.geometry.centroid.x,row.geometry.centroid.y);x,y=int(x),int(y)
    x0=max(0,x-55);y0=max(0,y-55);x1=min(data.shape[2],x+55);y1=min(data.shape[1],y+55)
    rgb=data[:3,y0:y1,x0:x1].transpose(1,2,0)
    ax.imshow(rgb)
    parts=list(row.geometry.geoms) if hasattr(row.geometry,'geoms') else [row.geometry]
    for part in parts:
        pts=np.array([inv*p for p in part.exterior.coords]);ax.plot(pts[:,0]-x0,pts[:,1]-y0,color='red',lw=1)
    ax.set_xlim(0,x1-x0);ax.set_ylim(y1-y0,0);ax.set_title('Maleza indicada: '+str(row.id));ax.axis('off')
    mask=geometry_mask([row.geometry],rgb.shape[:2],tr*Affine.translation(x0,y0),invert=True)
    pix=rgb[mask].astype(float);exg=(2*pix[:,1]-pix[:,0]-pix[:,2])/(pix.sum(axis=1)+1)
    stats.append({'id':int(row.id),'area_m2':row.geometry.area,'pixels':len(pix),'exg_median':float(np.median(exg))})
fig.tight_layout();fig.savefig(out/'muestras_nuevas_revision.png',dpi=120)
(out/'muestras_nuevas_resumen.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats))
