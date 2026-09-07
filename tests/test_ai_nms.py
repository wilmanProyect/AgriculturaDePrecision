# -*- coding: utf-8 -*-
"""
Pruebas unitarias para la Supresión de No-Máximos (NMS) del Motor IA.
"""

from ai.inference.detection_result import Detection
from ai.utils.nms import non_max_suppression


def _make_detection(class_id, confidence, bbox):
    x1, y1, x2, y2 = bbox
    return Detection(
        class_id=class_id,
        class_name=f"clase_{class_id}",
        confidence=confidence,
        bbox_pixel=bbox,
        center_pixel=((x1 + x2) / 2, (y1 + y2) / 2)
    )


def test_nms_removes_overlapping_duplicates():
    """Dos cajas de la misma clase muy solapadas deben fusionarse, quedando la de mayor confianza."""
    high_conf = _make_detection(0, 0.9, (10, 10, 50, 50))
    low_conf_duplicate = _make_detection(0, 0.6, (12, 12, 52, 52))

    result = non_max_suppression([high_conf, low_conf_duplicate], iou_threshold=0.45)

    assert len(result) == 1
    assert result[0] is high_conf


def test_nms_keeps_non_overlapping_boxes():
    """Cajas que no se solapan deben conservarse todas."""
    box_a = _make_detection(0, 0.8, (0, 0, 10, 10))
    box_b = _make_detection(0, 0.7, (100, 100, 110, 110))

    result = non_max_suppression([box_a, box_b], iou_threshold=0.45)

    assert len(result) == 2


def test_nms_does_not_suppress_across_classes():
    """Cajas solapadas de clases distintas (ej. planta vs árbol) no deben suprimirse entre sí."""
    plant = _make_detection(0, 0.9, (10, 10, 50, 50))
    tree = _make_detection(1, 0.85, (10, 10, 50, 50))

    result = non_max_suppression([plant, tree], iou_threshold=0.45)

    assert len(result) == 2


def test_nms_empty_list_returns_empty():
    assert non_max_suppression([]) == []
