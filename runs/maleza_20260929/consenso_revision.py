from pathlib import Path
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes, geometry_mask
from scipy.ndimage import label, binary_closing
from shapely.geometry import shape
from shapely import wkt
from affine import Affine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
out=root/'supervisado_v4'
with rasterio.open(root/'supervisado_v3/puntuacion_maleza.tif') as src:
    p3=src.read(1);tr=src.transform
with rasterio.open(out/'puntuacion_maleza.tif') as src:p4=src.read(1)
boundary=max([wkt.loads(i['geometry']['wkt']) for i in json.loads((root/'referencia_usuario.json').read_text())['items']],key=lambda g:g.area)
mask=(p3>=98)&(p4>=99)&(p3<255)&(p4<255)
mask &= geometry_mask([boundary.buffer(-1)],mask.shape,tr,invert=True)
mask=binary_closing(mask,iterations=1)
labels,n=label(mask);sizes=np.bincount(labels.ravel());keep=sizes*.05**2>=.18;keep[0]=False
records=[]
for geo,idx in shapes(labels.astype('int32'),mask=keep[labels],transform=tr):
    geom=shape(geo).intersection(boundary)
    if geom.is_empty or geom.area>3:continue
    rect=np.array(geom.minimum_rotated_rectangle.exterior.coords)
    lengths=np.linalg.norm(np.diff(rect,axis=0),axis=1)
    elongation=lengths.max()/max(lengths.min(),.0001)
    if elongation>2.5 or geom.area/geom.convex_hull.area<.55:continue
    records.append({'id':len(records)+1,'area_m2':geom.area,'estado':'Candidata - verificar',
        'metodo':'Consenso color y textura','muestras_maleza':9,'muestras_soya':5,'geometry':geom})
gdf=gpd.GeoDataFrame(records,geometry='geometry',crs=32720)
gdf.to_file(out/'maleza_revision_conservadora.gpkg',layer='candidatos',driver='GPKG')
print('CANDIDATOS',len(gdf),'AREA',gdf.area.sum(),flush=True)
a=np.load(root/'supervisado_v3/training_data.npz');b=np.load(out/'training_data.npz')
assert np.array_equal(a['groups'],b['groups']) and np.array_equal(a['y'],b['y'])
pred=(a['oof']>=.98)&(b['oof']>=.99)
stats=[]
for group in np.unique(a['groups']):
    where=a['groups']==group
    stats.append({'group':int(group),'class_id':int(a['y'][where][0]),'positive_fraction':float(pred[where].mean())})
report={'candidates':len(gdf),'candidate_area_m2':gdf.area.sum(),
    'cross_validation_pixel_consensus_before_shape_filters':stats,
    'note':'Exploratory consensus; strict filter misses weeds. Not independently validated. Color-only and broad texture outputs rejected for soybean false positives.',
    'all_inside':bool(gdf.difference(boundary.buffer(.000001)).is_empty.all()),'all_valid':bool(gdf.is_valid.all())}
(out/'resumen_conservador.json').write_text(json.dumps(report,indent=2))
z=np.load(root/'crop.npz');data=z['data'];inv=~tr;H,W=mask.shape
fig,axes=plt.subplots(2,3,figsize=(15,10))
for ax,i in zip(axes.flat,np.linspace(0,len(gdf)-1,6).astype(int)):
    geom=gdf.iloc[i].geometry;x,y=inv*(geom.centroid.x,geom.centroid.y);x,y=int(x),int(y)
    x0=max(0,x-60);x1=min(W,x+60);y0=max(0,y-60);y1=min(H,y+60)
    ax.imshow(data[:3,y0:y1,x0:x1].transpose(1,2,0))
    for cand in gdf[gdf.distance(geom)<4].geometry:
        pts=np.array([inv*p for p in cand.exterior.coords]);ax.plot(pts[:,0]-x0,pts[:,1]-y0,'r-',lw=1)
    ax.set_xlim(0,x1-x0);ax.set_ylim(y1-y0,0);ax.axis('off');ax.set_title('Candidato '+str(gdf.iloc[i]['id']))
fig.tight_layout();fig.savefig(out/'revision_conservadora.png',dpi=120)
