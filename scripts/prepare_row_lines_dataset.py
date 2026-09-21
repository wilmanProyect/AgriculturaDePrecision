# -*- coding: utf-8 -*-
"""
Prepara fotos crudas de dron (DJI, con EXIF/XMP de altitud y distancia focal)
para etiquetar líneas de siembra en Roboflow: las re-muestrea a la resolución
real que usa el plugin en producción (evita el desfase de escala que ya
afectó a los modelos anteriores) y sub-muestrea el solape entre fotos
consecutivas de un mismo vuelo.

No sube nada a Roboflow ni entrena: solo prepara los archivos de imagen.

Uso:
    python scripts/prepare_row_lines_dataset.py \
        "C:\\Users\\VICTUS\\Downloads\\Imagenes para entrenamiento" \
        data/images/_row_lines_raw_resampled \
        --resolution-m 0.05 --stride 4

Cada subcarpeta de origen (un vuelo) se procesa por separado y el nombre de
vuelo queda en el nombre del archivo de salida, para poder separar
train/valid/test por vuelo (no al azar) al etiquetar.
"""
import argparse
import os
import re
from pathlib import Path

from PIL import Image

# Sensor RGB del DJI Mavic 3 Multispectral (M3M). Ajustar si el dron es otro.
_SENSOR_WIDTH_MM = 17.3


def _read_xmp(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read(200_000)
    start = raw.find(b"<x:xmpmeta")
    end = raw.find(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        return ""
    return raw[start:end + 12].decode("utf-8", errors="ignore")


def _xmp_float(xmp: str, key: str):
    match = re.search(rf'{key}="([+-]?[\d.]+)"', xmp)
    return float(match.group(1)) if match else None


def estimate_gsd_m(path: str) -> float:
    """Estima metros/píxel a partir del EXIF (distancia focal) y la altitud relativa (XMP)."""
    img = Image.open(path)
    exif = img._getexif() or {}
    focal_mm = exif.get(37386) or exif.get(41989)  # FocalLength / FocalLengthIn35mm (fallback)
    if focal_mm is None:
        raise ValueError(f"Sin distancia focal en EXIF: {path}")
    focal_mm = float(focal_mm)

    xmp = _read_xmp(path)
    altitude_m = _xmp_float(xmp, "RelativeAltitude") or _xmp_float(xmp, "drone-dji:RelativeAltitude")
    if altitude_m is None:
        raise ValueError(f"Sin altitud relativa (XMP RelativeAltitude) en: {path}")

    width_px = img.size[0]
    # GSD (m/px) = ancho_sensor_mm * altura_m / (focal_mm * ancho_img_px); focal y sensor en mm,
    # se cancelan las unidades de longitud y queda directamente en metros/píxel.
    return (_SENSOR_WIDTH_MM * altitude_m) / (focal_mm * width_px)


def resample_to_resolution(src_path: str, dst_path: str, target_resolution_m: float) -> None:
    native_gsd_m = estimate_gsd_m(src_path)
    img = Image.open(src_path)
    scale = native_gsd_m / target_resolution_m  # <1 si hay que achicar la imagen
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    resized = img.resize(new_size, Image.LANCZOS)
    resized.save(dst_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Carpeta con subcarpetas por vuelo (cada una con fotos .JPG)")
    parser.add_argument("output_dir", help="Carpeta de salida")
    parser.add_argument("--resolution-m", type=float, default=0.05,
                         help="Resolución objetivo en metros/píxel (debe coincidir con RowOptions.resolution_m)")
    parser.add_argument("--stride", type=int, default=4,
                         help="Usar 1 de cada N fotos por vuelo, para reducir solape redundante")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flights = [d for d in sorted(input_dir.iterdir()) if d.is_dir()]
    total_in, total_out = 0, 0
    for flight_dir in flights:
        images = sorted(p for p in flight_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
        selected = images[::args.stride]
        total_in += len(images)
        print(f"{flight_dir.name}: {len(images)} fotos -> {len(selected)} seleccionadas (stride={args.stride})")
        for src in selected:
            dst = output_dir / f"{flight_dir.name}__{src.stem}.jpg"
            try:
                resample_to_resolution(str(src), str(dst), args.resolution_m)
                total_out += 1
            except ValueError as exc:
                print(f"  omitida ({exc})")
    print(f"\nListo: {total_out} imágenes preparadas en {output_dir} (de {total_in} originales).")
    print("Subilas a Roboflow como proyecto de Instance Segmentation, separando por el prefijo de vuelo "
          "(antes de '__') entre train/valid/test.")


if __name__ == "__main__":
    main()
