"""Detección RGB de surcos y vacíos en metros, independiente de QGIS.

La señal de vegetación está separada del seguimiento geométrico para permitir
incorporar posteriormente máscaras de un modelo de segmentación validado.
"""
from dataclasses import dataclass, replace
import math
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window, from_bounds
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, label, binary_fill_holes
from scipy.signal import find_peaks
from shapely.geometry import LineString, box

from .vegetation_index import VegetationIndexCalculator


class RowAnalysisCanceled(Exception):
    """Cancelación cooperativa; no publicar resultados parciales como completos."""


@dataclass(frozen=True)
class RowOptions:
    # None = detectar automáticamente la resolución nativa del ortomosaico (ver
    # `resolve_resolution_m`). Es el valor recomendado: usa todo el detalle real
    # disponible sin exigir que el usuario conozca la resolución del vuelo de
    # antemano. Ya no se achica según el área de las parcelas (ver max_pixels).
    resolution_m: Optional[float] = None
    spacing_m: float = 0.45
    min_gap_m: float = 1.0
    vegetation_threshold: float = 0.065
    min_row_m: float = 5.0
    # Presupuesto de píxeles POR PIEZA (no por parcela completa): lotes grandes
    # (ej. 150 ha) se dividen en piezas que respeten este límite a la resolución
    # elegida (ver `split_large_geometries`), en vez de fallar o perder detalle
    # coarseando la resolución. Subir este valor da piezas más grandes (menos
    # fragmentación de líneas en los bordes) a cambio de más memoria por pieza.
    max_pixels: int = 12_000_000

    def validate(self):
        values = (self.spacing_m, self.min_gap_m, self.min_row_m)
        if self.resolution_m is not None:
            values = values + (self.resolution_m,)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Resolución, separación y longitudes deben ser positivas y finitas.")
        if self.resolution_m is not None and self.spacing_m < 5 * self.resolution_m:
            raise ValueError("Usa al menos cinco píxeles por separación entre surcos.")
        if not math.isfinite(self.vegetation_threshold) or not -1 <= self.vegetation_threshold <= 2:
            raise ValueError("Umbral de vegetación fuera de rango.")


def _check_cancel(callback):
    if callback and callback():
        raise RowAnalysisCanceled()


def native_resolution_m(raster_path: str, crs) -> float:
    """
    Resolución nativa aproximada del ortomosaico (metros/píxel) en `crs`, sin
    forzar ningún remuestreo: usa el cálculo por defecto de rasterio
    (`calculate_default_transform` sin `resolution=`), que preserva
    aproximadamente la cantidad de píxeles de origen.
    """
    with rasterio.open(raster_path) as src:
        grid, _, _ = calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds)
    return abs(grid.a)


def resolve_resolution_m(options: RowOptions, raster_path: str, parcels_metric: gpd.GeoDataFrame) -> RowOptions:
    """
    Si `options.resolution_m` es `None`, lo fija a la resolución nativa del
    ortomosaico (no fabrica detalle que no existe). Ya no lo achica según el área
    de las parcelas: lotes grandes se dividen en piezas más chicas
    (`split_large_geometries`) para respetar `max_pixels` sin perder detalle real.
    Devuelve `options` sin cambios si ya trae un valor fijo.
    """
    if options.resolution_m is not None:
        return options
    native = native_resolution_m(raster_path, parcels_metric.crs)
    return replace(options, resolution_m=native)


# El recorte real de cada pieza se redondea hacia afuera a límites de píxel
# (`_crop`/`_crop_orthomosaic_rgb`), así que puede terminar un poco más grande
# que el área teórica de la pieza. Este margen evita que ese redondeo haga
# fallar el chequeo de `max_pixels` justo en el borde.
_TILE_SAFETY_MARGIN = 0.85


def tile_area_m2(options: RowOptions) -> float:
    """Área máxima por pieza (m²) para `split_large_geometries`, dado `options`."""
    return options.max_pixels * _TILE_SAFETY_MARGIN * options.resolution_m ** 2


def split_large_geometries(metric: gpd.GeoDataFrame, max_area_m2: float):
    """
    Divide cada geometría de `metric` cuya área supere `max_area_m2` en una
    grilla de piezas más chicas (intersecadas con su forma real), para poder
    analizar lotes grandes (ej. 150 ha) sin necesitar un único recorte gigante
    en memoria. Devuelve una lista de `(etiqueta, geometría)`: la etiqueta es
    "<n>" para una geometría sin dividir, o "<n>.<k>" para la pieza k de la
    geometría n original — mantiene trazabilidad al lote real.

    Una hilera que cruza el límite entre dos piezas queda partida en el
    resultado (más líneas, más cortas) — no afecta la suma total de metros
    sembrados/no sembrados, que es lo que reporta el resumen de la corrida.
    """
    pieces = []
    for i, geom in enumerate(metric.geometry):
        label = str(i + 1)
        if geom is None or geom.is_empty or geom.area <= max_area_m2:
            pieces.append((label, geom))
            continue
        minx, miny, maxx, maxy = geom.bounds
        tile_side = math.sqrt(max_area_m2)
        n_cols = max(1, math.ceil((maxx - minx) / tile_side))
        n_rows = max(1, math.ceil((maxy - miny) / tile_side))
        col_w = (maxx - minx) / n_cols
        row_h = (maxy - miny) / n_rows
        tile_idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                cell = box(minx + c * col_w, miny + r * row_h,
                           minx + (c + 1) * col_w, miny + (r + 1) * row_h)
                piece = geom.intersection(cell)
                if piece.is_empty:
                    continue
                tile_idx += 1
                pieces.append((f"{label}.{tile_idx}", piece))
    return pieces


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _frame(records, crs, gap=False):
    columns = ['parcela_id', 'surco_id', 'sector', 'long_m', 'estado', 'geometry']
    if gap:
        columns.insert(-1, 'umbral_m')
    return gpd.GeoDataFrame(records, columns=columns, geometry='geometry', crs=crs)


class RowDetector:
    """Un resultado preliminar: baja vegetación no equivale a falla confirmada."""

    def __init__(self, options=None):
        self.options = options or RowOptions()
        self.options.validate()

    def analyze(self, raster_path, parcels, progress=None, should_stop=None, crs=None):
        """
        `crs` es opcional: si el llamador ya reprojectó `parcels` a un CRS métrico
        (recomendado desde un hilo en segundo plano de QGIS, ver
        `plugin_qgis.dialog`), pasarlo evita volver a tocar
        `GeoDataFrame.estimate_utm_crs()`/`to_crs()` acá — en algunos entornos
        ese cálculo puede colgar el proceso con un crash nativo de PROJ (access
        violation) si es la primera vez que se toca un CRS de pyproj desde ese
        hilo. Sin `crs`, se calcula igual que antes (para uso fuera de QGIS).
        """
        if parcels.empty or parcels.crs is None:
            raise ValueError("Selecciona parcelas con geometría y CRS definidos.")
        if crs is not None:
            metric = parcels
        else:
            crs = parcels.estimate_utm_crs()
            if crs is None:
                raise ValueError("No se pudo determinar un CRS métrico para las parcelas.")
            metric = parcels.to_crs(crs)
        self.options = resolve_resolution_m(self.options, raster_path, metric)
        for i, geom in enumerate(metric.geometry):
            if geom is None or geom.is_empty or geom.geom_type not in ('Polygon', 'MultiPolygon'):
                raise ValueError(f"Parcela {i + 1}: geometría poligonal vacía o inválida.")
            if not geom.is_valid:
                raise ValueError(f"Parcela {i + 1}: repara su geometría antes de analizar.")

        # Lotes grandes (ej. 150 ha) se dividen en piezas que respeten max_pixels a
        # la resolución elegida, para no necesitar un único recorte gigante en
        # memoria. Una hilera que cruce el límite entre piezas queda partida en el
        # resultado (más líneas, más cortas) — no afecta la suma total de metros.
        pieces = split_large_geometries(metric, tile_area_m2(self.options))

        rows, gaps, warnings = [], [], []
        with rasterio.open(raster_path) as src:
            if src.crs is None or src.count < 3:
                raise ValueError("Se requiere un ortomosaico RGB con CRS definido.")
            colors = src.colorinterp
            rgb = [colors.index(c) + 1 for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)] if all(
                c in colors for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)) else [1, 2, 3]
            grid, grid_width, grid_height = calculate_default_transform(
                src.crs, crs, src.width, src.height, *src.bounds, resolution=self.options.resolution_m)
            with WarpedVRT(src, crs=crs, transform=grid, width=grid_width, height=grid_height,
                           resampling=Resampling.bilinear, add_alpha=ColorInterp.alpha not in colors) as vrt:
                total = len(pieces)
                for idx, (label, geom) in enumerate(pieces, start=1):
                    _check_cancel(should_stop)
                    if geom is None or geom.is_empty or geom.geom_type not in ('Polygon', 'MultiPolygon'):
                        continue  # pieza degenerada del recorte en grilla, no un error del usuario
                    crop = self._crop(vrt, rgb, geom)
                    if crop is None:
                        warnings.append(f"Parcela {label}: sin cobertura RGB válida.")
                        continue
                    signal, valid, transform = crop
                    rr, gg = self.trace_signal(signal, valid, transform, geom, label, should_stop)
                    offset = len(rows)
                    for record in rr + gg:
                        record['surco_id'] += offset
                    rows.extend(rr)
                    gaps.extend(gg)
                    if not rr:
                        warnings.append(f"Parcela {label}: no se identificaron surcos con suficiente evidencia.")
                    if progress:
                        progress(100 * idx / total)
        _check_cancel(should_stop)
        return {'rows_gdf': _frame(rows, crs), 'gaps_gdf': _frame(gaps, crs, True),
                'warnings': warnings, 'method': 'RGB', 'options': self.options}

    def _crop(self, vrt, bands, polygon):
        from rasterio.errors import WindowError
        try:
            requested = from_bounds(*polygon.bounds, transform=vrt.transform)
            left, top = math.floor(requested.col_off), math.floor(requested.row_off)
            window = Window(left, top, math.ceil(requested.col_off + requested.width)-left,
                            math.ceil(requested.row_off + requested.height)-top)
            window = window.intersection(Window(0, 0, vrt.width, vrt.height))
        except WindowError:
            return None
        # Explicit output resolution bounds memory even when source GSD is finer.
        res = self.options.resolution_m
        bounds = rasterio.windows.bounds(window, vrt.transform)
        width = max(1, math.ceil((bounds[2] - bounds[0]) / res))
        height = max(1, math.ceil((bounds[3] - bounds[1]) / res))
        if width * height > self.options.max_pixels:
            raise ValueError("Parcela demasiado grande para esta resolución. Divide la parcela o aumenta la resolución en metros/píxel.")
        transform = vrt.window_transform(window) * Affine.scale(window.width / width, window.height / height)
        data = vrt.read(bands, window=window, out_shape=(3, height, width), masked=True,
                        out_dtype='float32', resampling=Resampling.bilinear)
        valid = ~np.ma.getmaskarray(data).any(axis=0)
        if ColorInterp.alpha in vrt.colorinterp:
            alpha = vrt.read(vrt.colorinterp.index(ColorInterp.alpha) + 1, window=window,
                             out_shape=(height, width), resampling=Resampling.nearest)
            valid &= alpha > 0
        valid &= geometry_mask([polygon], (height, width), transform, invert=True)
        signal = VegetationIndexCalculator.compute_exg(*data.filled(0))
        valid &= np.isfinite(signal)
        return (np.nan_to_num(signal).astype('float32'), valid, transform) if valid.any() else None

    def trace_signal(self, signal, valid, transform, polygon, parcel_id='1', should_stop=None):
        """Follow a normalized vegetation signal; separate from raster I/O/model inference."""
        _check_cancel(should_stop)
        res = self.options.resolution_m
        smooth = gaussian_filter(np.where(valid, signal, 0), 1)
        dy, dx = np.gradient(smooth)
        sigma = max(2, self.options.spacing_m / res)
        jxx = gaussian_filter(dx * dx, sigma)
        jyy = gaussian_filter(dy * dy, sigma)
        jxy = gaussian_filter(dx * dy, sigma)
        energy = np.sqrt((jxx-jyy)**2 + 4*jxy*jxy)
        orientation = (0.5*np.arctan2(2*jxy, jxx-jyy) + np.pi/2) % np.pi
        hist = np.bincount(np.minimum((orientation[valid]*180/np.pi).astype(int), 179),
                           weights=energy[valid], minlength=180)
        hist = gaussian_filter1d(hist, 2, mode='wrap')
        if hist.max() <= 1e-8:
            return [], []
        angles = []
        work = hist.copy()
        for _ in range(3):
            peak = int(work.argmax())
            if work[peak] < hist.max() * .18:
                break
            angles.append(peak)
            distance = np.abs((np.arange(180)-peak+90) % 180-90)
            work[distance < 25] = 0
        distances = np.stack([np.abs((np.rad2deg(orientation)-a+90) % 180-90) for a in angles])
        labels = distances.argmin(axis=0)
        rows, gaps = [], []
        for sector, angle in enumerate(angles, 1):
            _check_cancel(should_stop)
            # Smooth evidence across interruptions; unknown/NoData remains excluded.
            evidence = gaussian_filter(energy * (labels == sector-1), sigma*2)
            total = gaussian_filter(energy, sigma*2)
            region = valid & (evidence > .55*np.maximum(total, 1e-12))
            components, count = label(region)
            sizes = np.bincount(components.ravel())
            keep = sizes >= max(100, 4*self.options.min_row_m*self.options.spacing_m/res**2)
            keep[0] = False
            region = binary_fill_holes(keep[components]) & valid
            rr, gg = self._trace_direction(signal, region, transform, polygon, angle,
                                            parcel_id, sector, len(rows), should_stop)
            rows.extend(rr)
            gaps.extend(gg)
        return rows, gaps

    def _trace_direction(self, signal, region, transform, polygon, angle, parcel_id, sector, offset, stop):
        opt = self.options
        res = opt.resolution_m
        a = np.deg2rad(angle)
        c, s = np.cos(a), np.sin(a)
        ys, xs = np.nonzero(region)
        if len(xs) < 100:
            return [], []
        u = c*xs+s*ys
        v = -s*xs+c*ys
        u0, v0 = np.floor([u.min(), v.min()])
        width, height = np.ceil([u.max()-u0+1, v.max()-v0+1]).astype(int)
        if width*height > opt.max_pixels*2:
            raise ValueError("Sector demasiado grande; divide la parcela para analizarla.")
        yy, xx = np.indices((height, width), dtype=np.float32)
        sx, sy = c*(xx+u0)-s*(yy+v0), s*(xx+u0)+c*(yy+v0)
        im = map_coordinates(signal, [sy, sx], order=1, mode='constant')
        inside = map_coordinates(region.astype('uint8'), [sy, sx], order=0, mode='constant') > 0
        del yy, xx, sx, sy
        ridge = gaussian_filter(im, (1, 4)) - gaussian_filter(im, (max(2, opt.spacing_m/res/2), 4))
        # Refine the integer-degree tensor estimate using coherent ridge projections.
        mid = width/2
        xgrid = np.arange(0, width, max(4, int(.5/res)))
        ygrid = np.arange(height)
        best = None
        for slope in np.linspace(-.035, .035, 15):
            _check_cancel(stop)
            Y = ygrid[:, None] + slope*(xgrid[None, :]-mid)
            X = np.broadcast_to(xgrid, Y.shape)
            mask = map_coordinates(inside.astype('uint8'), [Y, X], order=0, mode='constant')
            vals = map_coordinates(ridge, [Y, X], order=1, mode='constant')
            profile = (vals*mask).sum(axis=1)/np.maximum(mask.sum(axis=1), 1)
            score = np.mean(profile**2)
            if best is None or score > best[0]:
                best = score, slope, profile
        _, slope, profile = best
        peaks, _ = find_peaks(profile, distance=max(3, int(.75*opt.spacing_m/res)), prominence=.004)
        rows, gaps = [], []
        for index, seed in enumerate(peaks):
            _check_cancel(stop)
            # Disjoint corridors prevent adjacent tracks from merging or crossing.
            separation = min(seed-peaks[index-1] if index else opt.spacing_m/res,
                             peaks[index+1]-seed if index+1 < len(peaks) else opt.spacing_m/res)
            radius = max(1, min(int(.4*separation), int(.22/res)))
            offsets = np.arange(-radius, radius+1)
            centers = np.arange(0, width, max(1, int(.5/res)))
            if len(centers) < 2:
                continue
            Y = seed+slope*(centers[:, None]-mid)+offsets
            X = np.broadcast_to(centers[:, None], Y.shape)
            reward = map_coordinates(ridge, [Y, X], order=1, mode='constant')
            allowed = map_coordinates(inside.astype('uint8'), [Y, X], order=0, mode='constant')
            reward = reward*allowed-.0008*offsets[None, :]**2
            score = reward[0].copy()
            penalty = .008*(offsets[:, None]-offsets[None, :])**2
            back = []
            for k, r in enumerate(reward[1:]):
                if k % 100 == 0:
                    _check_cancel(stop)
                matrix = score[:, None]-penalty
                prev = matrix.argmax(axis=0)
                back.append(prev)
                score = matrix[prev, np.arange(len(offsets))]+r
            states = [int(score.argmax())]
            for prev in back[::-1]:
                states.append(int(prev[states[-1]]))
            path_y = gaussian_filter1d(Y[np.arange(len(centers)), states[::-1]], 2)
            px = np.arange(centers[-1]+1)
            py = np.interp(px, centers, path_y)
            ok = map_coordinates(inside.astype('uint8'), [py, px], order=0, mode='constant') > 0
            strength = map_coordinates(ridge, [py, px], order=1, mode='constant')
            green = np.max([map_coordinates(im, [py+d, px], order=1, mode='constant')
                            for d in (-radius/2, 0, radius/2)], axis=0)
            green = gaussian_filter1d(green, 2)
            for start, end in _runs(ok):
                if (end-start)*res < opt.min_row_m or np.mean(strength[start:end] > .006) < .2:
                    continue
                cx = c*(px[start:end]+u0)-s*(py[start:end]+v0)+.5
                cy = s*(px[start:end]+u0)+c*(py[start:end]+v0)+.5
                coords = list(zip(transform.a*cx+transform.b*cy+transform.c,
                                  transform.d*cx+transform.e*cy+transform.f))
                line = LineString(coords).intersection(polygon)
                parts = list(line.geoms) if hasattr(line, 'geoms') else [line]
                for part in parts:
                    if part.geom_type != 'LineString' or part.length < opt.min_row_m:
                        continue
                    row_id = offset+len(rows)+1
                    base = dict(parcela_id=parcel_id, surco_id=row_id, sector=sector)
                    rows.append(dict(base, long_m=part.length, estado='Surco estimado', geometry=part))
                    low = green[start:end] < opt.vegetation_threshold
                    for gs, ge in _runs(low):
                        # Require vegetation at both ends; no invented gaps at sector/NoData edges.
                        if gs == 0 or ge == len(low) or ge-gs < 2:
                            continue
                        gap = LineString(coords[gs:ge]).intersection(part)
                        # Intersecting identical polylines may split at every vertex.
                        if gap.geom_type == 'MultiLineString':
                            from shapely.ops import linemerge
                            gap = linemerge(gap)
                        segments = list(gap.geoms) if hasattr(gap, 'geoms') else [gap]
                        for segment in segments:
                            if segment.geom_type == 'LineString' and segment.length >= opt.min_gap_m:
                                gaps.append(dict(base, long_m=segment.length, umbral_m=opt.min_gap_m,
                                                 estado='Posible falla - verificar', geometry=segment))
        return rows, gaps
