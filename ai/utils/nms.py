# -*- coding: utf-8 -*-
"""
Supresión de No-Máximos (NMS) para fusionar detecciones duplicadas
que aparecen en los solapes entre tiles del ortomosaico.
"""

from typing import List

import numpy as np

from ai.inference.detection_result import Detection


def non_max_suppression(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """
    Aplica NMS por clase sobre una lista de detecciones en coordenadas de píxel
    del ortomosaico completo, eliminando cajas duplicadas de bajo solape.
    """
    if not detections:
        return []

    kept: List[Detection] = []

    by_class = {}
    for detection in detections:
        by_class.setdefault(detection.class_id, []).append(detection)

    for class_detections in by_class.values():
        boxes = np.array([d.bbox_pixel for d in class_detections], dtype=np.float64)
        scores = np.array([d.confidence for d in class_detections], dtype=np.float64)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            kept.append(class_detections[i])

            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h
            union = areas[i] + areas[rest] - intersection
            iou = np.where(union > 0, intersection / union, 0.0)

            order = rest[iou <= iou_threshold]

    return kept
