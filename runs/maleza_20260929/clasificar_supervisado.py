"""Supervised exploratory weed/soy segmentation, no sowing layers as inputs."""
from pathlib import Path
import json
import os
import numpy as np
import geopandas as gpd
import joblib
from shapely import wkt
from shapely.geometry import shape
from affine import Affine
import rasterio
from rasterio.features import geometry_mask, rasterize, shapes
from scipy.ndimage import gaussian_filter, uniform_filter, label, binary_closing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONTEXT=os.environ.get('WEED_CONTEXT','0')=='1'
HALO=96 if CONTEXT else 32
OUT=Path(__file__).resolve().parent/('supervisado_v4' if CONTEXT else 'supervisado_v3')
OUT.mkdir(exist_ok=True)
ROOT=OUT.parent
z=np.load(ROOT/'crop.npz');rgba=z['data'];tr=Affine(*z['tr']);inv=~tr
H,W=rgba.shape[1:];RES=.05
boundary=max([wkt.loads(item['geometry']['wkt']) for item in json.loads((ROOT/'referencia_usuario.json').read_text())['items']],key=lambda g:g.area)
samples=[]
for filename,klass in [('muestras_nuevas.json',1),('muestras_soya.json',0)]:
    source=json.loads((ROOT/filename).read_text())['items']
    frame=gpd.GeoDataFrame({'class_id':[klass]*len(source),'source_fid':[i['id'] for i in source]},
        geometry=[wkt.loads(i['geometry']['wkt']) for i in source],crs=4326).to_crs(32720)
    frame.geometry=frame.geometry.make_valid()
    samples.extend(frame.to_dict('records'))
sample_gdf=gpd.GeoDataFrame(samples,geometry='geometry',crs=32720)
sample_gdf['sample_id']=np.arange(len(samples))
sample_gdf.to_file(OUT/'muestras_usuario.gpkg',layer='muestras',driver='GPKG')

def features(rgb):
    rgb=rgb.astype('float32')/255
    r,g,b=rgb;total=r+g+b+1e-6
    exg=(2*g-r-b)/total
    bright=total/3
    values=[r,g,b,r/total,b/total,exg,bright,(np.max(rgb,axis=0)-np.min(rgb,axis=0))/(np.max(rgb,axis=0)+1e-6)]
    for sigma in (1,3,6):
        for channel in (exg,bright):
            mean=gaussian_filter(channel,sigma)
            std=np.sqrt(np.maximum(gaussian_filter(channel*channel,sigma)-mean*mean,0))
            values.extend([mean,std])
    dy,dx=np.gradient(gaussian_filter(exg,1))
    jxx=gaussian_filter(dx*dx,5);jyy=gaussian_filter(dy*dy,5);jxy=gaussian_filter(dx*dy,5)
    values.append(np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(jxx+jyy+1e-6))
    if CONTEXT:
        for sigma in (10,20):
            a=gaussian_filter(dx*dx,sigma); b2=gaussian_filter(dy*dy,sigma); cross=gaussian_filter(dx*dy,sigma)
            values.append(np.sqrt((a-b2)**2+4*cross*cross)/(a+b2+1e-6))
            values.append(gaussian_filter((exg>.12).astype('float32'),sigma))
            local=gaussian_filter(exg,sigma)
            values.extend([local,exg-local,bright-gaussian_filter(bright,sigma)])
    return np.stack(values,axis=-1),exg

rng=np.random.default_rng(42)
XX=[];YY=[];GG=[];sample_counts=[]
for index,row in sample_gdf.iterrows():
    pts=np.array([inv*(x,y) for x,y in [(row.geometry.bounds[0],row.geometry.bounds[1]),
        (row.geometry.bounds[0],row.geometry.bounds[3]),(row.geometry.bounds[2],row.geometry.bounds[1]),
        (row.geometry.bounds[2],row.geometry.bounds[3])]])
    x0=max(0,int(pts[:,0].min())-HALO);x1=min(W,int(pts[:,0].max())+HALO+1)
    y0=max(0,int(pts[:,1].min())-HALO);y1=min(H,int(pts[:,1].max())+HALO+1)
    f,exg=features(rgba[:3,y0:y1,x0:x1])
    interior=row.geometry.buffer(-.025)
    if interior.is_empty:interior=row.geometry
    mask=geometry_mask([interior],f.shape[:2],tr*Affine.translation(x0,y0),invert=True)
    mask&=(exg>.06)&(rgba[3,y0:y1,x0:x1]>250)
    values=f[mask]
    if len(values)>600:values=values[rng.choice(len(values),600,replace=False)]
    if len(values)<10:raise ValueError(f'Muestra {index}: pocos pixeles de vegetacion')
    XX.append(values);YY.extend([int(row.class_id)]*len(values));GG.extend([index]*len(values))
    sample_counts.append({'sample_id':index,'class_id':int(row.class_id),'pixels':len(values)})
X=np.concatenate(XX);y=np.array(YY);groups=np.array(GG)

def fit(train):
    model=RandomForestClassifier(n_estimators=120,max_depth=12,min_samples_leaf=8,
        max_features=.7,n_jobs=4,class_weight='balanced',random_state=42)
    _,inverse,count=np.unique(groups[train],return_inverse=True,return_counts=True)
    weights=1/count[inverse];weights*=len(weights)/weights.sum()
    model.fit(X[train],y[train],sample_weight=weights)
    return model

oof=np.zeros(len(y))
for train,test in LeaveOneGroupOut().split(X,y,groups):
    oof[test]=fit(train).predict_proba(X[test])[:,1]

def metrics(threshold):
    stats=[]
    for group in np.unique(groups):
        chosen=groups==group
        stats.append((int(y[chosen][0]),float(np.mean(oof[chosen]>=threshold))))
    return {'threshold':float(threshold),'weed_recall_mean_polygon':float(np.mean([v for k,v in stats if k==1])),
            'soy_false_positive_mean_polygon':float(np.mean([v for k,v in stats if k==0])),
            'soy_false_positive_worst_polygon':float(max([v for k,v in stats if k==0]))}

curve=[metrics(t) for t in np.arange(.6,.981,.02)]
acceptable=[v for v in curve if v['soy_false_positive_mean_polygon']<=.03 and v['soy_false_positive_worst_polygon']<=.10]
chosen=max(acceptable,key=lambda v:v['weed_recall_mean_polygon']) if acceptable else curve[-1]
threshold=max(.80,chosen['threshold'])
chosen=metrics(threshold)
report={'samples':sample_counts,'validation':'Leave-one-polygon-out; nearby polygons still spatially correlated',
    'validation_auc_pixels':float(roc_auc_score(y,oof)),'selected_threshold':chosen,'threshold_curve':curve,
    'method':'Random Forest RGB + multiscale texture, user-labelled weeds and soybean',
    'inputs':['result.tif','poligono id=1','maleza (9 user polygons)','Soya (5 user polygons)'],
    'limitations':['Small geographically clustered training sample; not independent field validation.',
    'Tree votes are not calibrated probabilities. No YOLO model was trained.',
    'Vegetation pixels only; masked soil/NoData not classified. No sowing line/failure inputs.']}
(OUT/'validacion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'validation':chosen,'auc':report['validation_auc_pixels']}),flush=True)
np.savez_compressed(OUT/'training_data.npz',X=X,y=y,groups=groups,oof=oof)
model=fit(np.arange(len(y)))
joblib.dump({'model':model,'threshold':threshold,'feature_version':2 if CONTEXT else 1,'resolution_m':RES},OUT/'clasificador.joblib')
if not acceptable or chosen['weed_recall_mean_polygon']<.35:
    report['status']='REJECTED: insufficient discrimination on held-out polygons'
    (OUT/'validacion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    raise SystemExit('No publicar: discriminacion insuficiente en validacion')

valid=geometry_mask([boundary],(H,W),tr,invert=True)&(rgba[3]>250)
probability=np.zeros((H,W),dtype='float32')
for y0 in range(0,H,256):
    y1=min(H,y0+256)
    for x0 in range(0,W,512):
        x1=min(W,x0+512);xa=max(0,x0-HALO);xb=min(W,x1+HALO);ya=max(0,y0-HALO);yb=min(H,y1+HALO)
        f,exg=features(rgba[:3,ya:yb,xa:xb])
        f=f[y0-ya:y1-ya,x0-xa:x1-xa]; exg=exg[y0-ya:y1-ya,x0-xa:x1-xa]
        mask=valid[y0:y1,x0:x1]&(exg>.06)
        if mask.any(): probability[y0:y1,x0:x1][mask]=model.predict_proba(f[mask])[:,1]
    print(f'Filas {y1}/{H}',flush=True)
smooth=gaussian_filter(probability,.7)
mask=binary_closing(smooth>=threshold,iterations=1)&valid
components,n=label(mask)
sizes=np.bincount(components.ravel())
keep=sizes*RES**2>=.06;keep[0]=False
mask=keep[components]
sums=np.bincount(components.ravel(),weights=probability.ravel(),minlength=n+1)
records=[]
for geom,idx in shapes(components.astype('int32'),mask=mask,transform=tr):
    gg=shape(geom).intersection(boundary)
    if gg.is_empty:continue
    records.append({'id':len(records)+1,'area_m2':gg.area,'voto_modelo':float(sums[int(idx)]/sizes[int(idx)]),
        'estado':'Maleza candidata - revisar','metodo':'Supervisado RGB textura','geometry':gg})
gdf=gpd.GeoDataFrame(records,columns=['id','area_m2','voto_modelo','estado','metodo','geometry'],geometry='geometry',crs=32720)
if len(gdf):gdf.to_file(OUT/'maleza_supervisada.gpkg',layer='maleza_candidata',driver='GPKG')
with rasterio.open(OUT/'puntuacion_maleza.tif','w',driver='GTiff',height=H,width=W,count=1,dtype='uint8',
    crs='EPSG:32720',transform=tr,nodata=255,compress='deflate',tiled=True) as dst:
    dst.write(np.where(valid,np.round(probability*100),255).astype('uint8'),1)
report.update({'candidates':len(gdf),'candidate_area_m2':float(gdf.area.sum()),'valid_geometries':bool(gdf.is_valid.all()),
    'all_inside':bool(gdf.difference(boundary.buffer(.000001)).is_empty.all())})
(OUT/'validacion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['candidates','candidate_area_m2','valid_geometries','all_inside']}),flush=True)
# Visual review includes both classes and geographically spread predictions.
items=[('Maleza anotada',sample_gdf.iloc[i].geometry) for i in (0,4,8)]
items += [('Soya anotada',sample_gdf.iloc[i].geometry) for i in (9,11,13)]
if len(gdf):items += [('Prediccion fuera de muestras',gdf.iloc[i].geometry) for i in np.linspace(0,len(gdf)-1,3).astype(int)]
fig,axes=plt.subplots(3,3,figsize=(15,15))
for ax,(title,geom) in zip(axes.flat,items):
    x,yc=inv*(geom.centroid.x,geom.centroid.y);x,yc=int(x),int(yc)
    x0=max(0,x-50);x1=min(W,x+50);y0=max(0,yc-50);y1=min(H,yc+50)
    ax.imshow(rgba[:3,y0:y1,x0:x1].transpose(1,2,0));ax.set_title(title)
    for candidate in gdf[gdf.distance(geom)<4].geometry:
        parts=list(candidate.geoms) if hasattr(candidate,'geoms') else [candidate]
        for part in parts:
            pts=np.array([inv*p for p in part.exterior.coords]);ax.plot(pts[:,0]-x0,pts[:,1]-y0,color='red',lw=1)
    ax.set_xlim(0,x1-x0);ax.set_ylim(y1-y0,0);ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'revision_visual.png',dpi=120)
