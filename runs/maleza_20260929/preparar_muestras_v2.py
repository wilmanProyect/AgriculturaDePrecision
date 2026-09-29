"""Preserve user labels and native-detail image chips for a two-class classifier."""
from pathlib import Path
import json
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from PIL import Image

root=Path(__file__).resolve().parent
out=root/'dataset_revision_v2'
out.mkdir(exist_ok=True)
samples=gpd.read_file(root/'muestras_maleza_usuario_v2.gpkg')
manifest=[]
with rasterio.open('C:/Users/VICTUS/Downloads/result.tif') as src:
    for _,row in samples.iterrows():
        # 5 x 5 m surrounding context at approximately native GSD.
        center=row.geometry.centroid
        transform=from_origin(center.x-2.5,center.y+2.5,.025,.025)
        rgb=np.zeros((3,200,200),dtype='uint8')
        for b in range(3):
            reproject(rasterio.band(src,b+1),rgb[b],src_transform=src.transform,src_crs=src.crs,
                      dst_transform=transform,dst_crs='EPSG:32720',resampling=Resampling.bilinear)
        # 255 = unlabelled; surrounding plants are NOT automatically treated as soy.
        mask=np.full((200,200),255,dtype='uint8')
        mask[geometry_mask([row.geometry],mask.shape,transform,invert=True)]=1
        stem=f'maleza_{int(row.id):02d}'
        Image.fromarray(rgb.transpose(1,2,0)).save(out/(stem+'.png'))
        Image.fromarray(mask).save(out/(stem+'_mask.png'))
        manifest.append({'image':stem+'.png','mask':stem+'_mask.png','user_id':int(row.id),
                         'class':1,'class_name':'maleza','crs':'EPSG:32720',
                         'transform':list(transform)[:6],'source':'maleza: anotacion del usuario'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
(out/'LEEME.txt').write_text('Muestras confirmadas de maleza indicadas por el usuario.\n'
    'Mascara: 1=maleza, 255=sin etiquetar. El fondo NO es soya confirmada.\n'
    'Faltan ejemplos negativos de soya para evaluar falsos positivos.\n'
    'La resolucion de salida 0.025 m no agrega detalle al original (~0.0266 m).\n'
    'Separar validacion por grupos espaciales; no dividir pixeles contiguos al azar.\n',encoding='utf-8')
print(f'{len(manifest)} recortes y mascaras guardados en {out}')
