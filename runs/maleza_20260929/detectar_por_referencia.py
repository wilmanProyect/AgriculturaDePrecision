"""Reference-color candidate segmentation. Only raster and live polygon geometries.
Does not read sowing lines, failures, or previous weed candidates.
"""
from pathlib import Path
import json
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import shape
from affine import Affine
from rasterio.features import geometry_mask, shapes
from scipy.ndimage import gaussian_filter, label, binary_closing, binary_fill_holes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
source=json.loads((OUT/'referencia_usuario.json').read_text())
geoms=[wkt.loads(item['geometry']['wkt']) for item in source['items']]
boundary=max(geoms,key=lambda g:g.area)
reference=min(geoms,key=lambda g:g.area)
z=np.load(OUT/'crop.npz'); data=z['data']; tr=Affine(*z['tr']); inv=~tr
H,W=data.shape[1:]; resolution=.05
valid=geometry_mask([boundary],(H,W),tr,invert=True)&(data[3]>250)
sample=geometry_mask([reference.buffer(-.05)],(H,W),tr,invert=True)&valid
# Smooth one pixel to reduce compression/shadow noise while retaining plant detail.
rgb=np.stack([gaussian_filter(b.astype('float32'),.8) for b in data[:3]])
total=rgb.sum(axis=0)+1
rn=rgb[0]/total; bn=rgb[2]/total
brightness=total/765
exg=(2*rgb[1]-rgb[0]-rgb[2])/total
# Use sunlit vegetation inside the user-confirmed weed polygon, not soil/shadow.
sample &= exg>.06
sample &= brightness>=np.percentile(brightness[sample],35)
features=np.stack([rn[sample],bn[sample],brightness[sample]],axis=1)
mean=features.mean(axis=0)
cov=np.cov(features,rowvar=False)+np.diag([.00001,.00001,.00005])
precision=np.linalg.inv(cov)
diff=features-mean
sample_d=np.einsum('ij,jk,ik->i',diff,precision,diff)
cutoff=float(np.percentile(sample_d,90))
distance=np.zeros((H,W),dtype='float32')
for start in range(0,H,128):
    end=min(H,start+128)
    f=np.stack([rn[start:end],bn[start:end],brightness[start:end]],axis=-1)-mean
    distance[start:end]=np.einsum('...i,ij,...j->...',f,precision,f)
raw=valid&(exg>.06)&(distance<=cutoff)
mask=binary_fill_holes(binary_closing(raw,iterations=1))&valid
components,n=label(mask)
sizes=np.bincount(components.ravel())
# Require spatial support and limit large merged canopy objects (not single weeds).
min_area=max(.04,reference.area*.15)
max_area=max(1.5,reference.area*5)
keep=(sizes*resolution**2>=min_area)&(sizes*resolution**2<=max_area);keep[0]=False
mask=keep[components]
score_sum=np.bincount(components.ravel(),weights=distance.ravel(),minlength=n+1)
records=[]
for geometry,component in shapes(components.astype('int32'),mask=mask,transform=tr):
    geom=shape(geometry).intersection(boundary)
    if geom.is_empty:continue
    rect=np.array(geom.minimum_rotated_rectangle.exterior.coords)
    lengths=np.linalg.norm(np.diff(rect,axis=0),axis=1)
    if lengths.max()/max(lengths.min(),.001)>4:continue
    idx=int(component)
    records.append({'id':len(records)+1,'area_m2':geom.area,'dist_color':float(score_sum[idx]/sizes[idx]),
                    'referencia':2,'estado':'Candidata por similitud - revisar','metodo':'RGB muestra usuario',
                    'geometry':geom})
gdf=gpd.GeoDataFrame(records,geometry='geometry',crs=32720)
gdf.to_file(OUT/'maleza_por_referencia.gpkg',layer='maleza_candidata',driver='GPKG')
gpd.GeoDataFrame({'clase':['Maleza indicada por usuario'],'id':[2]},geometry=[reference],crs=32720).to_file(
    OUT/'muestra_confirmada.gpkg',layer='referencia_usuario',driver='GPKG')
report={'candidates':len(gdf),'candidate_area_m2':float(gdf.area.sum()),'reference_area_m2':reference.area,
        'training_pixels':int(sample.sum()),'color_mean':mean.tolist(),'color_distance_threshold':cutoff,
        'min_area_m2':min_area,'max_area_m2':max_area,
        'reference_recovered':bool(gdf.intersects(reference).any()),
        'all_inside':bool(gdf.difference(boundary.buffer(.000001)).is_empty.all()),
        'valid_geometries':bool(gdf.is_valid.all()),
        'inputs':['result.tif','poligono id=1 (limite)','poligono id=2 (maleza confirmada por usuario)'],
        'method':'Similitud de color normalizado y brillo; segmentacion de manchas. No usa capas de siembra.',
        'limitations':'Una sola muestra; similitud no es probabilidad de maleza. No identifica especie ni garantiza distinguir toda la soya.'}
(OUT/'resumen_referencia.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report),flush=True)
# Inspect reference plus spatially distributed candidates.
selected=[reference]+[gdf.iloc[i].geometry for i in np.linspace(0,len(gdf)-1,5).astype(int)]
fig,axes=plt.subplots(2,3,figsize=(15,10))
for ax,geom in zip(axes.flat,selected):
    x,y=inv*(geom.centroid.x,geom.centroid.y);x,y=int(x),int(y)
    x0=max(0,x-50);x1=min(W,x+50);y0=max(0,y-50);y1=min(H,y+50)
    ax.imshow(data[:3,y0:y1,x0:x1].transpose(1,2,0))
    for candidate in gdf[gdf.distance(geom)<4].geometry:
        pts=np.array([inv*p for p in candidate.exterior.coords]);ax.plot(pts[:,0]-x0,pts[:,1]-y0,color='red',lw=1)
    ax.set_xlim(0,x1-x0);ax.set_ylim(y1-y0,0);ax.axis('off')
axes.flat[0].set_title('Referencia del usuario y segmentacion')
fig.tight_layout();fig.savefig(OUT/'revision_referencia.png',dpi=130)
