#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Punto de entrada principal para el motor de Agricultura de Precisión.
Ejecuta la lógica de procesamiento (GIS, IA, Reportes) de forma autónoma.
"""

import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="🌱 Sistema de Agricultura de Precisión - Consola de Ejecución"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="run",
        choices=["run", "train", "infer", "report", "all"],
        help="Modo de ejecución del sistema"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Ruta al archivo de configuración"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[infer] Ruta al ortomosaico (GeoTIFF) sobre el que detectar plantas"
    )
    return parser.parse_args()

def _load_config(config_path: str) -> dict:
    import yaml
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _run_train(ai_config: dict) -> None:
    from ai.training import DatasetConfig, TrainingConfig, YOLOTrainer

    training_cfg = ai_config.get("training", {})
    model_cfg = ai_config.get("model", {})
    classes = list(ai_config.get("classes", {"0": "planta"}).values())

    dataset = DatasetConfig(training_cfg.get("dataset_dir", "data/images"), classes)
    dataset.validate()

    trainer = YOLOTrainer(TrainingConfig(
        data_yaml=training_cfg.get("data_yaml", "data/images/data.yaml"),
        model_arch=model_cfg.get("architecture", "yolo11m.pt"),
        epochs=training_cfg.get("epochs", 100),
        imgsz=model_cfg.get("imgsz", 640),
        batch=training_cfg.get("batch", 16),
        device=model_cfg.get("device", "cpu"),
        patience=training_cfg.get("patience", 50),
        project=training_cfg.get("project", "data/models"),
        name=training_cfg.get("name", "plant_detector")
    ))
    trainer.train()

def _run_infer(ai_config: dict, image_path: str) -> None:
    from ai.inference import PlantDetector
    from core.gis.raster.raster_manager import RasterManager

    model_cfg = ai_config.get("model", {})
    inference_cfg = ai_config.get("inference", {})

    detector = PlantDetector(
        weights_path=model_cfg.get("weights_path", "data/models/plant_detector/weights/best.pt"),
        device=model_cfg.get("device", "cpu"),
        conf_threshold=model_cfg.get("conf_threshold", 0.25),
        iou_threshold=model_cfg.get("iou_threshold", 0.45),
        imgsz=model_cfg.get("imgsz", 640)
    )
    detector.load_model()

    with RasterManager() as raster_manager:
        raster_manager.open(image_path)
        detections = detector.predict_orthomosaic(
            raster_manager,
            tile_size=inference_cfg.get("tile_size", 1024),
            overlap=inference_cfg.get("overlap", 0.2)
        )

    counts = PlantDetector.count_by_class(detections)
    print(f"Detecciones por clase: {counts}")

def main():
    args = parse_args()
    print("==================================================")
    print("  🌱 SISTEMA DE AGRICULTURA DE PRECISIÓN (MOTOR)  ")
    print("==================================================")
    print(f"Modo seleccionado: {args.mode}")
    print(f"Configuración:     {args.config}")
    print("--------------------------------------------------")

    ai_config = _load_config(args.config).get("ai", {})

    try:
        if args.mode == "train":
            print("[AI] Iniciando fase de entrenamiento de YOLO11...")
            _run_train(ai_config)
        elif args.mode == "infer":
            if not args.image:
                print("Error: el modo 'infer' requiere --image <ruta al ortomosaico>.")
                sys.exit(1)
            print("[AI/GIS] Iniciando inferencia en imágenes y georreferenciación...")
            _run_infer(ai_config, args.image)
        elif args.mode == "report":
            print("[CORE] Generando análisis de densidad y reportes (PDF/Excel)...")
        elif args.mode == "all":
            print("[CORE/AI] Ejecutando flujo completo del sistema...")
        else:
            print("Ejecutando proceso base. Use --help para ver las opciones disponibles.")
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        sys.exit(1)

    print("\nProceso finalizado correctamente.")

if __name__ == "__main__":
    main()
