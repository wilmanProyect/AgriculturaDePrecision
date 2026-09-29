"""Exploratory inter-row vegetation map; NOT a validated weed classifier."""
from pathlib import Path
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask, rasterize, shapes
from affine import Affine
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_opening, label
from shapely.geometry import shape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
RASTER='C:/Users/VICTUS/Downloads/result.tif'
POLYGON='C:/Users/VICTUS/Documents/poligono.shp'
ROWS='C:/Users/VICTUS/Documents/Agroptima/AnalisisSiembra/20260917_174940_d1a3b8/lineas_siembra.gpkg'
poly=gpd.read_file(POLYGON).geometry.iloc[0]
rows=gpd.read_file(ROWS)
res=.05
cache=OUT/'crop.npz'
if not cache.exists():
    a=np.deg2rad(23.5); u=np.array([np.cos(a),np.sin(a)]); v=np.array([-np.sin(a),np.cos(a)])
    origin=np.array(poly.centroid.coords[0]); coords=np.array(poly.exterior.coords)-origin
    lo=np.floor(np.array([(coords@u).min(),(coords@v).min()])/res)*res-1
    hi=np.ceil(np.array([(coords@u).max(),(coords@v).max()])/res)*res+1
    w,h=np.ceil((hi-lo)/res).astype(int); start=origin+lo[0]*u+hi[1]*v
    tr=Affine(res*u[0],-res*v[0],start[0],res*u[1],-res*v[1],start[1])
    data=np.zeros((4,h,w),np.uint8)
    with rasterio.open(RASTER) as src:
        for b in range(4):
            reproject(rasterio.band(src,b+1),data[b],src_transform=src.transform,src_crs=src.crs,
                      dst_transform=tr,dst_crs='EPSG:32720',resampling=Resampling.bilinear,num_threads=4)
    np.savez_compressed(cache,data=data,tr=np.array(tr)[:6])
else:
    z=np.load(cache); data=z['data']; tr=Affine(*z['tr'])
h,w=data.shape[1:]
valid=geometry_mask([poly.buffer(-1)],(h,w),tr,invert=True)&(data[3]>250)
r,g,b=data[:3].astype('float32')
exg=(2*g-r-b)/(r+g+b+1)
vegetation=valid&(exg>.12)&(g>r*1.08)&(g>b*1.08)
line=rasterize(((geom,1) for geom in rows.geometry),out_shape=(h,w),transform=tr,dtype='uint8')
distance=distance_transform_edt(line==0,sampling=res).astype('float32')
# Only inspect vegetation 18-30 cm from existing row centers, not unmodelled parts.
# Exclude closed canopy: crop and weeds cannot be separated there with this heuristic.
coverage=gaussian_filter(vegetation.astype('float32'),sigma=10)
eligible=valid&(distance>=.18)&(distance<=.30)&(coverage<.60)
# Evaluate entire vegetation components, avoiding thin cutouts of crop leaf edges.
components,n=label(vegetation)
sizes=np.bincount(components.ravel())
outside=np.bincount(components.ravel(),weights=(distance>.14).ravel(),minlength=n+1)
supported=np.bincount(components.ravel(),weights=eligible.ravel(),minlength=n+1)
keep=(sizes*res*res>=.03)&(sizes*res*res<=1.5)&(outside/np.maximum(sizes,1)>.95)&(supported/np.maximum(sizes,1)>.60)
keep[0]=False
suspect=keep[components]
records=[]
for geom,value in shapes(suspect.astype('uint8'),mask=suspect,transform=tr):
    gg=shape(geom).intersection(poly)
    if gg.is_empty or gg.area<.03: continue
    rectangle=np.array(gg.minimum_rotated_rectangle.exterior.coords)
    lengths=np.linalg.norm(np.diff(rectangle,axis=0),axis=1)
    if lengths.max()/max(lengths.min(),.001)>4: continue
    records.append({'id':len(records)+1,'area_m2':gg.area,'estado':'Vegetacion intersurco - revisar',
                    'metodo':'Vegetacion aislada + distancia','exg_min':.12,'dist_min_m':.14,
                    'dist_max_m':.30,'validado':0,'geometry':gg})
gdf=gpd.GeoDataFrame(records,columns=['id','area_m2','estado','metodo','exg_min','dist_min_m','dist_max_m','validado','geometry'],geometry='geometry',crs=32720)
suspect=rasterize(((geom,1) for geom in gdf.geometry),out_shape=(h,w),transform=tr,dtype='uint8').astype(bool) if len(gdf) else np.zeros((h,w),bool)
if len(gdf): gdf.to_file(OUT/'posible_maleza.gpkg',layer='vegetacion_intersurco',driver='GPKG')
report={'parcel_area_m2':poly.area,'candidate_patches':len(gdf),'flagged_vegetation_m2':float(gdf.area.sum()),
        'evaluated_interrow_m2':float(eligible.sum()*res*res),'resolution_m':res,
        'method':'Vegetacion RGB fuera de ejes de surcos existentes; sin modelo entrenado',
        'limitations':['No confirma maleza ni especie. Puede incluir hojas de soya y errores de los ejes.',
                        'No evalua maleza dentro del surco, dosel cerrado o zonas sin ejes cercanos.',
                        'El area marcada no representa la infestacion total del lote.'],
        'inputs':[RASTER,POLYGON,ROWS]}
(OUT/'resumen.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report),flush=True)
if not len(gdf): raise SystemExit('No candidates passing isolation filters')
fig,ax=plt.subplots(figsize=(20,6)); ax.imshow(data[:3,::4,::4].transpose(1,2,0));
overlay=np.zeros((h//4+(h%4>0),w//4+(w%4>0),4),dtype='float32'); overlay[suspect[::4,::4]]=[1,0,1,.9]
ax.imshow(overlay); ax.axis('off'); fig.tight_layout(); fig.savefig(OUT/'vista_general.png',dpi=160); plt.close(fig)
# Detailed evidence panels: sample the largest and median patches, not only best-looking areas.
ranked=gdf.sort_values('area_m2',ascending=False)
indices=[0,min(5,len(gdf)-1),len(gdf)//2]
fig,axes=plt.subplots(3,2,figsize=(12,16))
inv=~tr
for k,idx in enumerate(indices):
    item=ranked.iloc[idx]; cx,cy=inv*(item.geometry.centroid.x,item.geometry.centroid.y)
    x0=max(0,int(cx)-80); x1=min(w,int(cx)+80); y0=max(0,int(cy)-80); y1=min(h,int(cy)+80)
    rgb=data[:3,y0:y1,x0:x1].transpose(1,2,0)
    for col in range(2): axes[k,col].imshow(rgb); axes[k,col].axis('off')
    axes[k,0].set_title(f"Imagen RGB: candidato {item['id']}")
    sub=suspect[y0:y1,x0:x1]; layer=np.zeros((*sub.shape,4)); layer[sub]=[1,0,1,.65]
    axes[k,1].imshow(layer); axes[k,1].set_title('Vegetacion fuera del eje: verificar')
fig.tight_layout();fig.savefig(OUT/'revision_detalle.png',dpi=130)
