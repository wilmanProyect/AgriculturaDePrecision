"""Bounded recalibration experiment; never publish predictions on failed validation."""
from pathlib import Path
root=Path(__file__).resolve().parent
source=(root/'clasificar_supervisado.py').read_text()
source=source.split('\nvalid=geometry_mask')[0]
source=source.replace("CONTEXT=os.environ.get('WEED_CONTEXT','0')=='1'",'CONTEXT=True')
source=source.replace("OUT=Path(__file__).resolve().parent/('supervisado_v4' if CONTEXT else 'supervisado_v3')","OUT=Path(__file__).resolve().parent/'supervisado_v5_local'")
source=source.replace("source=json.loads((ROOT/filename).read_text())['items']", "source=json.loads((ROOT/filename).read_text())['items']\n    if klass==1: source += json.loads((ROOT/'prueba_local/malezas_2_usuario.json').read_text())['items']")
source=source.replace("'maleza (9 user polygons)'", "'maleza (9 previous + 14 new user polygons)'")
source=source.replace("'Small geographically clustered training sample; not independent field validation.'", "'Small spatially clustered sample; leave-one-polygon-out is exploratory, not independent field validation. New labels already used for model development.'")
exec(compile(source,str(root/'clasificar_supervisado.py'),'exec'))
