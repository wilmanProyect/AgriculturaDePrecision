# -*- coding: utf-8 -*-
"""
Entrena un modelo YOLO11 de SEGMENTACIÓN de instancias (yolo11m-seg) para detectar
líneas de siembra y líneas/tramos no sembrados directamente sobre ortomosaicos.

A diferencia de scripts/train_datasets.py (detección de plantas, cajas), este
entrena un modelo de segmentación: las etiquetas del dataset deben ser POLÍGONOS,
no bounding boxes, para poder seguir el trazo real de cada línea (incluidos los
giros semicirculares en las cabeceras, donde la sembradora da la vuelta).

Dataset esperado (formato YOLO-seg de Roboflow/Ultralytics):
    data/images/roboflow_row_lines/
        data.yaml
        train/images, train/labels
        valid/images, valid/labels
        test/images,  test/labels

Convención de clases (ver docs/lineas_siembra.md): usa estos nombres exactos en
data.yaml para que ai/inference/row_segmenter.py clasifique bien cada instancia
en "sembrado" o "no sembrado":
    names: ['linea_sembrada', 'linea_no_sembrada']

Al etiquetar: dibuja cada línea (incluidas las curvas de cabecera) como UN solo
polígono continuo que sigue su forma real, no como segmentos rectos separados.
Evita anotar el mismo tramo dos veces con clases distintas.

Orientación: en el ortomosaico las hileras pueden aparecer en cualquier ángulo
(diagonales, curvas, cruces), aunque las fotos de origen del dron muestren casi
siempre la misma orientación. Para que el modelo no aprenda a asociar "línea de
siembra" con una orientación fija (fue justo lo que rompió al modelo de pose
anterior), este script activa rotación aleatoria completa (`degrees=180`) y
espejado vertical (`flipud=0.5`) durante el entrenamiento — se suma al espejado
horizontal que Ultralytics ya trae activado por defecto. Esto cubre la
ORIENTACIÓN; no reemplaza tener ejemplos reales de curvas y cruces etiquetados
en el dataset, que es forma, no orientación (ver docs/lineas_siembra.md).

Uso:
    python scripts/train_row_lines.py
"""

from ai.training import TrainingConfig, YOLOTrainer

DATA_YAML = "data/images/roboflow_row_lines/data.yaml"


def main() -> None:
    trainer = YOLOTrainer(TrainingConfig(
        data_yaml=DATA_YAML,
        model_arch="yolo11m-seg.pt",
        epochs=150,          # las líneas finas y curvas tardan más en converger que cajas de plantas
        imgsz=960,           # más resolución que el detector de plantas (640): las líneas son delgadas
        batch=-1,            # autobatch: se ajusta a la VRAM disponible (RTX 2050 = poca VRAM)
        device="cuda",
        patience=50,
        project="data/models",
        name="row_lines_seg",
        workers=2,
        degrees=180,         # invarianza a la orientación: la hilera puede apuntar a cualquier ángulo
        flipud=0.5,          # complementa el espejado horizontal (fliplr=0.5, ya viene por defecto)
    ))
    trainer.train()
    trainer.validate()


if __name__ == "__main__":
    main()
