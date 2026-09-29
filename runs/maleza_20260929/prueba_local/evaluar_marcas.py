from pathlib import Path
import json
import geopandas as gpd
from shapely import wkt
P=Path(__file__).resolve().parent
items=json.loads((P/'malezas_2_usuario.json').read_text())['items']
g=gpd.GeoDataFrame({'fid_usuario':[i['id'] for i in items]},geometry=[wkt.loads(i['geometry']['wkt']) for i in items],crs=4326).to_crs(32720)
g.geometry=g.geometry.make_valid()
boundary=gpd.read_file(P/'limite.gpkg').geometry.union_all()
pred=gpd.read_file(P/'candidatos_prueba.gpkg')
union=pred.geometry.union_all()
g.geometry=g.geometry.intersection(boundary)
g=g[~g.geometry.is_empty].copy()
g['area_m2']=g.area
g['coincide_m2']=g.geometry.intersection(union).area
g['cobertura_pct']=100*g.coincide_m2/g.area_m2
g['resultado']=g.cobertura_pct.apply(lambda x:'Omitida' if x==0 else 'Parcial' if x<50 else 'Mayormente cubierta')
g.to_file(P/'evaluacion_malezas_2.gpkg',driver='GPKG')
g.drop(columns='geometry').to_csv(P/'evaluacion_malezas_2.csv',index=False)
truth=g.geometry.union_all()
r={'marked_total':len(items),'marked_inside':len(g),'any_overlap':int((g.coincide_m2>0).sum()),'at_least_half_covered':int((g.cobertura_pct>=50).sum()),'completely_missed':int((g.coincide_m2==0).sum()),'marked_area_m2':truth.area,'covered_marked_area_m2':truth.intersection(union).area,'marked_area_coverage_pct':100*truth.intersection(union).area/truth.area,'candidates_overlapping_marks':int(pred.intersects(truth).sum()),'candidate_count':len(pred),'note':'Independent of these new labels: existing predictions frozen before labels. Polygon area coverage, not plant precision. Unlabelled areas are unknown, not negatives.'}
(P/'evaluacion_malezas_2.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r));print(g.drop(columns='geometry').to_string(index=False))
