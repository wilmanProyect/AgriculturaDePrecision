"""Run the existing RGB classifier only within the user's live test boundary."""
from pathlib import Path
import ast, json
import numpy as np
import geopandas as gpd
import joblib
from affine import Affine
from shapely import wkt
from shapely.geometry import shape
from rasterio.features import geometry_mask, shapes
import rasterio
from scipy.ndimage import gaussian_filter, binary_closing, label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'prueba_local'; OUT.mkdir(exist_ok=True)
items=json.loads((ROOT/'prueba_limite.json').read_text())['items']
limit=gpd.GeoSeries([wkt.loads(i['geometry']['wkt']) for i in items],crs=4326).to_crs(32720).make_valid().union_all()
assert not limit.is_empty and limit.area>0
gpd.GeoDataFrame({'id':[1]},geometry=[limit],crs=32720).to_file(OUT/'limite.gpkg',driver='GPKG')
z=np.load(ROOT/'crop.npz'); data=z['data']; base=Affine(*z['tr']); inverse=~base
pts=np.array([inverse*(x,y) for x in limit.bounds[::2] for y in limit.bounds[1::2]])
halo=96
x0=max(0,int(np.floor(pts[:,0].min()))-halo);x1=min(data.shape[2],int(np.ceil(pts[:,0].max()))+halo)
y0=max(0,int(np.floor(pts[:,1].min()))-halo);y1=min(data.shape[1],int(np.ceil(pts[:,1].max()))+halo)
assert x1>x0 and y1>y0, 'Boundary outside cached orthomosaic'
rgba=data[:,y0:y1,x0:x1]; del data
tr=base*Affine.translation(x0,y0);inv=~tr
inside=geometry_mask([limit],rgba.shape[1:],tr,invert=True)
valid=inside&(rgba[3]>250)
tree=ast.parse((ROOT/'clasificar_supervisado.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='features')
CONTEXT=True
exec(compile(ast.Module(body=[function],type_ignores=[]),'<existing feature extractor>','exec'))
model=joblib.load(ROOT/'supervisado_v4/clasificador.joblib')['model']
scores=np.zeros(inside.shape,dtype=np.float32)
H,W=inside.shape
for ya in range(0,H,256):
    yb=min(H,ya+256)
    for xa in range(0,W,512):
        xb=min(W,xa+512)
        if not valid[ya:yb,xa:xb].any():continue
        aa=max(0,xa-halo);bb=min(W,xb+halo);cc=max(0,ya-halo);dd=min(H,yb+halo)
        f,exg=features(rgba[:3,cc:dd,aa:bb])
        f=f[ya-cc:yb-cc,xa-aa:xb-aa];exg=exg[ya-cc:yb-cc,xa-aa:xb-aa]
        take=valid[ya:yb,xa:xb]&(exg>.06)
        if take.any():scores[ya:yb,xa:xb][take]=model.predict_proba(f[take])[:,1]
    print(f'Processed {yb}/{H}',flush=True)
mask=binary_closing(gaussian_filter(scores,.7)>=.98,iterations=1)&valid
components,n=label(mask);sizes=np.bincount(components.ravel());keep=sizes*.0025>=.06;keep[0]=False
mask=keep[components]
records=[]
for geometry,value in shapes(mask.astype('uint8'),mask=mask,transform=tr):
    geom=shape(geometry).intersection(limit)
    if not geom.is_empty:records.append({'id':len(records)+1,'area_m2':geom.area,'estado':'Ensayo sin validar','geometry':geom})
gdf=gpd.GeoDataFrame(records,columns=['id','area_m2','estado','geometry'],crs=32720)
gdf.to_file(OUT/'candidatos_prueba.gpkg',driver='GPKG')
report={'method':'Existing V4 Random Forest, RGB and texture; threshold 0.98; min area 0.06 m2',
        'boundary_area_m2':limit.area,'candidates':len(gdf),'candidate_area_m2':float(gdf.area.sum()),
        'all_inside':bool(gdf.difference(limit.buffer(1e-6)).is_empty.all()),'all_valid':bool(gdf.is_valid.all()),
        'valid_raster_fraction':float(valid.sum()/inside.sum()),'samples_inside':{},
        'status':'UNVALIDATED experiment. Prior full-field result failed. Smaller boundary does not improve model.',
        'inputs':['crop.npz (orthomosaic RGB)','prueba_limite.json','supervisado_v4/clasificador.joblib']}
samples=[]
for fn,name in [('muestras_nuevas.json','maleza'),('muestras_soya.json','soya')]:
    it=json.loads((ROOT/fn).read_text())['items']
    gs=gpd.GeoSeries([wkt.loads(i['geometry']['wkt']) for i in it],crs=4326).to_crs(32720)
    overlap=gs.intersection(limit);report['samples_inside'][name]=int((overlap.area>0).sum())
    samples.extend([(name,g) for g in overlap if not g.is_empty])
(OUT/'informe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
with rasterio.open(OUT/'votos.tif','w',driver='GTiff',height=H,width=W,count=1,dtype='float32',crs=32720,transform=tr,nodata=-1,compress='deflate') as dst:
    dst.write(np.where(valid,scores,-1).astype('float32'),1)
fig,axes=plt.subplots(1,2,figsize=(20,11))
def draw(ax,geom,color,lw=1):
    for g in (geom.geoms if hasattr(geom,'geoms') else [geom]):
        if g.geom_type!='Polygon':continue
        xy=np.array([inv*p for p in g.exterior.coords]);ax.plot(xy[:,0],xy[:,1],color=color,lw=lw)
for ax in axes:
    ax.imshow(rgba[:3].transpose(1,2,0));draw(ax,limit,'yellow',1.5);ax.axis('off')
    for name,g in samples:draw(ax,g,'cyan' if name=='soya' else 'magenta',1.5)
for g in gdf.geometry:draw(axes[1],g,'red',.8)
axes[0].set_title('Ortomosaico; limite amarillo; soya cian; maleza magenta')
axes[1].set_title(f'{len(gdf)} candidatos del codigo existente - SIN VALIDAR')
fig.tight_layout();fig.savefig(OUT/'comparacion.png',dpi=130)
print(json.dumps(report),flush=True)
