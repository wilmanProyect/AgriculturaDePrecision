from detect_rows import *
from scipy.ndimage import gaussian_filter, maximum_filter1d

z=np.load(OUT/'crop.npz'); data=z['data']; exg=z['exg']; mask=z['mask']; tr=Affine(*z['tr'])
H,W=exg.shape
yy,xx=np.indices(exg.shape)
rows=[]; gaps=[]; diagnostics=[]

def trace(im,valid,zone,transpose=False):
    # Estimate the orientation from narrow vegetation ridges, removing broad canopy gradients.
    h,w=im.shape
    signal=gaussian_filter(im,(1,4))-gaussian_filter(im,(5,4))
    mid=w/2
    xs=np.arange(60,w-60,8)
    ys=np.arange(20,h-20)
    best=None
    for slope in np.arange(-.20,.201,.002):
        Y=ys[:,None]+slope*(xs[None,:]-mid)
        X=np.broadcast_to(xs,Y.shape)
        vals=map_coordinates(signal,[Y,X],order=1,mode='constant')
        ok=map_coordinates(valid.astype(float),[Y,X],order=0,mode='constant')
        profile=(vals*ok).sum(axis=1)/np.maximum(ok.sum(axis=1),1)
        score=np.mean(np.sort(profile**2)[-min(200,len(profile)):])
        if best is None or score>best[0]: best=(score,slope,profile)
    _,slope,prof=best
    peaks,props=find_peaks(prof,distance=7,prominence=.004)
    diagnostics.append({'zone':zone,'slope':float(slope),'seed_count':len(peaks),'median_spacing_m':float(np.median(np.diff(peaks))*.05)})
    # Dynamic programming follows each row within its own corridor, through vegetation gaps.
    for peak in peaks:
        seed=ys[peak]
        centers=np.arange(10,w-10,10)
        offsets=np.arange(-4,4.1,1)
        Y=seed+slope*(centers[:,None]-mid)+offsets
        X=np.broadcast_to(centers[:,None],Y.shape)
        reward=map_coordinates(signal,[Y,X],order=1,mode='constant')
        ok=map_coordinates(valid.astype(float),[Y,X],order=0,mode='constant')
        reward=reward*ok-.0008*offsets[None,:]**2
        score=reward[0].copy(); back=[]
        penalty=.008*(offsets[:,None]-offsets[None,:])**2
        for r in reward[1:]:
            matrix=score[:,None]-penalty
            prev=matrix.argmax(axis=0); back.append(prev); score=matrix[prev,np.arange(len(offsets))]+r
        states=[int(score.argmax())]
        for prev in back[::-1]: states.append(int(prev[states[-1]]))
        states=states[::-1]
        py=Y[np.arange(len(centers)),states]
        py=gaussian_filter1d(py,2)
        fine_x=np.arange(centers[0],centers[-1]+1)
        fine_y=np.interp(fine_x,centers,py)
        inside=map_coordinates(valid.astype(float),[fine_y,fine_x],order=0,mode='constant')>.5
        strength=map_coordinates(signal,[fine_y,fine_x],order=1,mode='constant')
        green=np.max([map_coordinates(im,[fine_y+d,fine_x],order=1,mode='constant') for d in [-2,0,2]],axis=0)
        green=gaussian_filter1d(green,2)
        edges=np.diff(np.r_[False,inside,False].astype(int)); starts=np.where(edges==1)[0]; ends=np.where(edges==-1)[0]
        for s,e in zip(starts,ends):
            if (e-s)*res<5 or np.mean(strength[s:e]>.006)<.2: continue
            px=fine_x[s:e]; py2=fine_y[s:e]
            if transpose: px,py2=py2,px
            coords=[tr*(float(x)+.5,float(y)+.5) for x,y in zip(px,py2)]
            geom=LineString(coords[::4]+[coords[-1]]).intersection(poly)
            if geom.is_empty: continue
            rowid=len(rows)+1
            rows.append({'surco_id':rowid,'sector':zone,'long_m':geom.length,'metodo':'crestas RGB; trazado automatico','geometry':geom})
            # Only internal low-vegetation stretches, bounded by observed vegetation.
            low=green[s:e]<.065
            ed=np.diff(np.r_[False,low,False].astype(int))
            for gs,ge in zip(np.where(ed==1)[0],np.where(ed==-1)[0]):
                if gs<10 or ge>len(low)-10 or (ge-gs)*res<1: continue
                gg=LineString(coords[gs:ge]).intersection(poly)
                if gg.is_empty or gg.length < 1: continue
                gaps.append({'surco_id':rowid,'sector':zone,'long_m':gg.length,'umbral_m':1.,'estado':'Posible falla - verificar','geometry':gg})

main=mask & (xx<2120+.14*yy)
head=mask & (xx>2160+.14*yy) & (xx<2540+.14*yy)
trace(exg,main,'Principal')
trace(exg.T,head.T,'Cabecera',True)
gdf=gpd.GeoDataFrame(rows,crs=32720); gf=gpd.GeoDataFrame(gaps,crs=32720)
gdf.to_file(OUT/'lineas_siembra.gpkg',layer='lineas_siembra',driver='GPKG')
if len(gf): gf.to_file(OUT/'posibles_fallas.gpkg',layer='posibles_fallas',driver='GPKG')
fig,ax=plt.subplots(figsize=(18,9)); ax.imshow(data.transpose(1,2,0)); inv=~tr
for collection,color,lw in [(rows,'cyan',.45),(gaps,'red',1.3)]:
    for f in collection:
        parts=list(f['geometry'].geoms) if hasattr(f['geometry'],'geoms') else [f['geometry']]
        for geom in parts:
            pts=np.array([inv*p for p in geom.coords]); ax.plot(pts[:,0],pts[:,1],color=color,lw=lw)
ax.set_xlim(0,W); ax.set_ylim(H,0); fig.tight_layout(); fig.savefig(OUT/'verificacion.png',dpi=170)
report={'rows':len(rows),'gaps':len(gaps),'row_length_m':float(gdf.length.sum()),'gap_length_m':float(gf.length.sum()) if len(gf) else 0,'diagnostics':diagnostics,'all_inside':bool(gdf.geometry.difference(poly.buffer(.00001)).is_empty.all()),'gap_threshold_m':1,'vegetation_threshold_exg':.065,'limitations':'RGB vegetation absence, not confirmed missing plants. Main and headland sectors separated visually; transition and road excluded. No individual plant counts.'}
(OUT/'resumen.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report))
