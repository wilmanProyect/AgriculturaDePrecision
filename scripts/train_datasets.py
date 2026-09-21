# -*- coding: utf-8 -*-
"""
Entrena un modelo YOLO11 independiente por cada dataset de data/images/.
Uso:
    python scripts/train_datasets.py                 # entrena los 5 datasets
    python scripts/train_datasets.py crop_weed        # entrena solo uno (por nombre)
"""

import sys

from ai.training import TrainingConfig, YOLOTrainer

DATASETS = {
    "crop_weed": "data/images/roboflow_crop_weed/data.yaml",
    "weed_crop_large": "data/images/roboflow_weed_crop/data.yaml",
    "plant_counting": "data/images/roboflow_plant_counting/data.yaml",
    "seedling": "data/images/roboflow_seedling/data.yaml",
    "weeding_detection": "data/images/roboflow_weeding_detection/data.yaml",
}


def train_one(name: str, data_yaml: str) -> None:
    print(f"\n=== Entrenando '{name}' ({data_yaml}) ===")
    trainer = YOLOTrainer(TrainingConfig(
        data_yaml=data_yaml,
        model_arch="yolo11m.pt",
        epochs=100,
        imgsz=640,
        batch=-1,           # autobatch: se ajusta a la VRAM disponible (RTX 2050 = poca VRAM)
        device="cuda",
        patience=50,
        project="data/models",
        name=name,
        workers=2,
    ))
    trainer.train()
    trainer.validate()


if __name__ == "__main__":
    selected = sys.argv[1:] or list(DATASETS.keys())
    for key in selected:
        if key not in DATASETS:
            print(f"Dataset desconocido: {key}. Opciones: {list(DATASETS.keys())}")
            sys.exit(1)
        train_one(key, DATASETS[key])
