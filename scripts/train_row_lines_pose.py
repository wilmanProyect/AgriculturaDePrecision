# -*- coding: utf-8 -*-
"""
Entrena un modelo YOLO11-POSE sobre el dataset "Crop-row" (Roboflow, formato
YOLOv8 con keypoints): cada línea es una caja + 2 keypoints (sus dos extremos
visibles en la foto). NO es el mismo tipo de modelo que scripts/train_row_lines.py
(ese es de segmentación/polígonos); este dataset viene con keypoints, no polígonos.

AVISO — dominio de las imágenes: este dataset son fotos a nivel de suelo (cámara
de robot/persona caminando entre las hileras), no vistas aéreas de dron. Un
modelo entrenado con esto probablemente NO generalice bien sobre un ortomosaico
(mismo problema que ya vimos con el modelo `weeding_detection`). Útil para
probar el pipeline de inferencia/huecos entre detecciones, pero antes de usarlo
como modelo final sobre el ortomosaico, valida sus resultados contra la realidad
del campo — o mejor, entrena con recortes reales de tus propios ortomosaicos.

Dataset ya extraído en: data/images/roboflow_row_lines_pose/

Uso:
    python scripts/train_row_lines_pose.py
"""

from ai.training import TrainingConfig, YOLOTrainer

DATA_YAML = "data/images/roboflow_row_lines_pose/data.yaml"


def main() -> None:
    trainer = YOLOTrainer(TrainingConfig(
        data_yaml=DATA_YAML,
        model_arch="yolo11m-pose.pt",
        epochs=150,
        imgsz=640,           # bajado de 960: yolo11m-pose a 960px no entra en 4GB (ver nota abajo)
        batch=4,              # fijo, sin autobatch: en la RTX 2050 (4GB) el propio sondeo de
                              # autobatch se quedó sin memoria y devolvió un batch imposible (16,
                              # "584% GPU"). Si igual da CUDA out of memory, bajá esto a 2, o
                              # cambiá model_arch a "yolo11s-pose.pt" (mismo dataset, modelo más chico).
        device="cuda",
        patience=50,
        project="data/models",
        name="row_lines_pose",
        workers=2,
    ))
    trainer.train()
    trainer.validate()


if __name__ == "__main__":
    main()
