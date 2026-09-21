from pathlib import Path
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from affine import Affine
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks
from shapely.geometry import LineString
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
poly=gpd.read_file('C:/Users/VICTUS/Documents/Poligono2.shp').geometry.iloc[0]
origin=np.array(poly.centroid.coords[0])
res=.05
def read_rotated(angle):
    a=np.deg2rad(angle); u=np.array([np.cos(a),np.sin(a)]); v=np.array([-np.sin(a),np.cos(a)])
    coords=np.array(poly.exterior.coords)-origin
    lo=np.floor(np.array([(coords@u).min(),(coords@v).min()])/res)*res-1
    hi=np.ceil(np.array([(coords@u).max(),(coords@v).max()])/res)*res+1
    w,h=np.ceil((hi-lo)/res).astype(int)
    start=origin+lo[0]*u+hi[1]*v
    tr=Affine(res*u[0],-res*v[0],start[0],res*u[1],-res*v[1],start[1])
    data=np.zeros((3,h,w),np.uint8)
    with rasterio.open('C:/Users/VICTUS/Downloads/result.tif') as src:
        for b in range(3):
            reproject(rasterio.band(src,b+1),data[b],src_transform=src.transform,src_crs=src.crs,dst_transform=tr,dst_crs='EPSG:32720',resampling=Resampling.bilinear,num_threads=4)
    mask=geometry_mask([poly],(h,w),tr,invert=True)
    rgb=data.astype(float); exg=(2*rgb[1]-rgb[0]-rgb[2])/(rgb.sum(axis=0)+1)
    return data,exg,mask,tr

if __name__=='__main__':
    data,exg,mask,tr=read_rotated(23)
    np.savez_compressed(OUT/'crop.npz',data=data,exg=exg,mask=mask,tr=np.array(tr)[:6])
    fig,ax=plt.subplots(figsize=(16,8)); ax.imshow(data.transpose(1,2,0)); ax.contour(mask,levels=[.5],colors='cyan'); fig.savefig(OUT/'crop.png',dpi=140); plt.close(fig)
    print(json.dumps({'shape':data.shape,'exg_percentiles':np.percentile(exg[mask],[10,25,50,75,90]).tolist()}))
