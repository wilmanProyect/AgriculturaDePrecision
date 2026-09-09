# -*- coding: utf-8 -*-
"""
Diálogo principal del plugin de Agricultura de Precisión.
Conecta la interfaz de QGIS con el Motor GIS (core) y el Motor IA (ai)
del proyecto AgriculturaDePrecision: detecta plantas con YOLO11, cuenta
por parcela, calcula densidad y colorea las parcelas automáticamente.
"""

import os

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsMapLayerProxyModel,
    QgsRendererCategory,
    QgsSymbol,
)
from qgis.gui import QgsFileWidget, QgsMapLayerComboBox
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .qgis_bridge import geodataframe_to_qgs_layer, layer_source_path
from .worker import DetectionTask, VegetationIndexTask

# Colores por clase de densidad: Alta (buena cobertura) -> verde, Baja -> rojo
_DENSITY_COLORS = {
    "Alta": QColor(27, 120, 55),
    "Media": QColor(255, 217, 47),
    "Baja": QColor(215, 48, 39),
    "Sin datos": QColor(189, 189, 189),
}

# Colores por clase de vigor vegetal (NDVI/NDRE): Vigorosa -> verde, Estresada -> naranja
_VEGETATION_COLORS = {
    "Vigorosa": QColor(26, 152, 80),
    "Moderada": QColor(166, 217, 106),
    "Estresada": QColor(253, 174, 97),
    "Suelo/Agua": QColor(140, 81, 10),
    "Sin datos": QColor(189, 189, 189),
}


class PrecisionAgDialog(QDialog):
    """Diálogo principal: detección de plantas, conteo, densidad, índices de vegetación y coloreado de parcelas."""

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._task = None
        self._last_result = None
        self._veg_task = None
        self._last_vegetation_result = None
        self._analysis_layer = None
        self._vegetation_layer = None

        self.setWindowTitle("Agricultura de Precisión")
        self.resize(600, 800)

        self._build_ui()
        self._load_defaults()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        inputs_group = QGroupBox("Entradas")
        form = QFormLayout()

        self.raster_combo = QgsMapLayerComboBox()
        self.raster_combo.setFilters(QgsMapLayerProxyModel.Filter.RasterLayer)
        form.addRow("Ortomosaico:", self.raster_combo)

        self.parcels_combo = QgsMapLayerComboBox()
        self.parcels_combo.setFilters(QgsMapLayerProxyModel.Filter.PolygonLayer)
        form.addRow("Parcelas:", self.parcels_combo)

        self.weights_widget = QgsFileWidget()
        self.weights_widget.setFilter("Modelos YOLO (*.pt)")
        form.addRow("Pesos del modelo (.pt):", self.weights_widget)

        inputs_group.setLayout(form)
        layout.addWidget(inputs_group)

        params_group = QGroupBox("Parámetros de inferencia")
        pform = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        pform.addRow("Device:", self.device_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        pform.addRow("Confianza mínima:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        pform.addRow("IoU (NMS):", self.iou_spin)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(128, 4096)
        self.tile_spin.setSingleStep(128)
        self.tile_spin.setValue(1024)
        pform.addRow("Tamaño de tile (px):", self.tile_spin)

        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 0.9)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setValue(0.2)
        pform.addRow("Solape entre tiles:", self.overlap_spin)

        params_group.setLayout(pform)
        layout.addWidget(params_group)

        actions_layout = QHBoxLayout()
        self.detect_btn = QPushButton("Detectar y Contar Plantas")
        self.detect_btn.clicked.connect(self.on_detect_clicked)
        actions_layout.addWidget(self.detect_btn)

        self.paint_btn = QPushButton("Pintar Parcelas por Densidad")
        self.paint_btn.clicked.connect(self.on_paint_clicked)
        self.paint_btn.setEnabled(False)
        actions_layout.addWidget(self.paint_btn)

        self.report_btn = QPushButton("Generar Reporte (CSV)")
        self.report_btn.clicked.connect(self.on_report_clicked)
        self.report_btn.setEnabled(False)
        actions_layout.addWidget(self.report_btn)

        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.clicked.connect(self.on_cancel_clicked)
        self.cancel_btn.setEnabled(False)
        actions_layout.addWidget(self.cancel_btn)
        layout.addLayout(actions_layout)

        veg_group = QGroupBox("Índice de Vegetación (NDVI / NDRE)")
        vform = QFormLayout()

        self.veg_raster_combo = QgsMapLayerComboBox()
        self.veg_raster_combo.setFilters(QgsMapLayerProxyModel.Filter.RasterLayer)
        vform.addRow("Ráster de índice:", self.veg_raster_combo)

        self.veg_index_combo = QComboBox()
        self.veg_index_combo.addItems(["ndvi", "ndre"])
        vform.addRow("Tipo de índice:", self.veg_index_combo)

        veg_group.setLayout(vform)
        layout.addWidget(veg_group)

        veg_actions_layout = QHBoxLayout()
        self.veg_calc_btn = QPushButton("Calcular Índice de Vegetación")
        self.veg_calc_btn.clicked.connect(self.on_vegetation_clicked)
        veg_actions_layout.addWidget(self.veg_calc_btn)

        self.veg_paint_btn = QPushButton("Pintar Parcelas por Vigor Vegetal")
        self.veg_paint_btn.clicked.connect(self.on_paint_vegetation_clicked)
        self.veg_paint_btn.setEnabled(False)
        veg_actions_layout.addWidget(self.veg_paint_btn)
        layout.addLayout(veg_actions_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("Resultados por parcela:"))
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["Parcela", "Área (m²)", "Plantas", "Densidad (pl/m²)", "Clase"]
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(500)
        self.log_output.setPlaceholderText("Estado de la operación...")
        layout.addWidget(self.log_output)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

    def _load_defaults(self) -> None:
        """Carga valores por defecto desde config/default.yaml del proyecto, si está disponible."""
        try:
            import core

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))
            config_path = os.path.join(project_root, "config", "default.yaml")

            from ai.utils.config_loader import load_ai_config

            ai_cfg = load_ai_config(config_path)
            model_cfg = ai_cfg.get("model", {})

            weights_default = model_cfg.get("weights_path")
            if weights_default:
                candidate = weights_default
                if not os.path.isabs(candidate):
                    candidate = os.path.join(project_root, candidate)
                if os.path.exists(candidate):
                    self.weights_widget.setFilePath(candidate)

            self.device_combo.setCurrentText(model_cfg.get("device", "cpu"))
            self.conf_spin.setValue(float(model_cfg.get("conf_threshold", 0.25)))
            self.iou_spin.setValue(float(model_cfg.get("iou_threshold", 0.45)))

            inference_cfg = ai_cfg.get("inference", {})
            self.tile_spin.setValue(int(inference_cfg.get("tile_size", 1024)))
            self.overlap_spin.setValue(float(inference_cfg.get("overlap", 0.2)))
        except Exception as e:
            self._log(f"No se pudo cargar config/default.yaml ({e}). Usando valores por defecto.")

    def _log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    # ------------------------------------------------------------ Detección

    def on_detect_clicked(self) -> None:
        raster_layer = self.raster_combo.currentLayer()
        parcels_layer = self.parcels_combo.currentLayer()

        if raster_layer is None or parcels_layer is None:
            QMessageBox.warning(self, "Faltan capas", "Selecciona un ortomosaico y una capa de parcelas.")
            return

        weights_path = self.weights_widget.filePath()
        if not weights_path or not os.path.exists(weights_path):
            QMessageBox.warning(self, "Modelo no encontrado", "Selecciona un archivo de pesos YOLO11 (.pt) válido.")
            return

        raster_path = layer_source_path(raster_layer)
        if not raster_path:
            QMessageBox.warning(
                self, "Ortomosaico no válido",
                "No se pudo resolver la ruta del archivo del ortomosaico "
                "(debe ser una capa respaldada por un archivo, ej. GeoTIFF)."
            )
            return

        try:
            from .qgis_bridge import qgs_vector_layer_to_geodataframe
            parcelas_gdf = qgs_vector_layer_to_geodataframe(parcels_layer)
        except Exception as e:
            QMessageBox.critical(self, "Error al leer parcelas", str(e))
            return

        if parcelas_gdf.crs is None:
            QMessageBox.warning(self, "CRS no definido", "La capa de parcelas no tiene un CRS definido.")
            return

        params = {
            "raster_path": raster_path,
            "parcelas_gdf": parcelas_gdf,
            "weights_path": weights_path,
            "device": self.device_combo.currentText(),
            "conf_threshold": self.conf_spin.value(),
            "iou_threshold": self.iou_spin.value(),
            "tile_size": self.tile_spin.value(),
            "overlap": self.overlap_spin.value(),
        }

        self._set_running(True)
        self._log("Iniciando detección de plantas...")

        self._task = DetectionTask(params)
        self._task.taskCompleted.connect(self._on_task_completed)
        self._task.taskTerminated.connect(self._on_task_terminated)
        self._task.progressChanged.connect(lambda p: self.progress_bar.setValue(int(p)))
        QgsApplication.taskManager().addTask(self._task)

    def _set_running(self, running: bool) -> None:
        self.detect_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.progress_bar.setValue(0)

    def on_cancel_clicked(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._log("Cancelando detección...")

    def _on_task_completed(self) -> None:
        self._set_running(False)
        result = self._task.result
        was_canceled = self._task.isCanceled()
        self._task = None

        self._last_result = result
        self.progress_bar.setValue(100)
        n_plantas = len(result["plantas_gdf"])
        if was_canceled:
            self._log(f"Detección cancelada. Resultados parciales: {n_plantas} plantas detectadas antes de cancelar.")
        else:
            self._log(f"Detección finalizada: {n_plantas} plantas detectadas.")

        self._populate_results_table(result["parcelas_gdf"], result["id_col"])

        try:
            geodataframe_to_qgs_layer(result["plantas_gdf"], "Plantas detectadas")
            self._analysis_layer = geodataframe_to_qgs_layer(result["parcelas_gdf"], "Parcelas - análisis")
            self._log("Capas 'Plantas detectadas' y 'Parcelas - análisis' añadidas al proyecto.")
        except Exception as e:
            self._log(f"No se pudieron crear las capas de resultado: {e}")
            self._analysis_layer = None

        self.paint_btn.setEnabled(self._analysis_layer is not None)
        self.report_btn.setEnabled(True)

    def _on_task_terminated(self) -> None:
        error = self._task.error if self._task is not None else None
        self._set_running(False)
        self._task = None

        if error is not None:
            self._log(f"Error: {error}")
            QMessageBox.critical(self, "Error durante la detección", str(error))
        else:
            self._log("La tarea de detección terminó de forma inesperada.")

    def _populate_results_table(self, parcelas_gdf, id_col: str) -> None:
        self.results_table.setRowCount(0)
        for _, row in parcelas_gdf.iterrows():
            r = self.results_table.rowCount()
            self.results_table.insertRow(r)
            self.results_table.setItem(r, 0, QTableWidgetItem(str(row[id_col])))
            self.results_table.setItem(r, 1, QTableWidgetItem(f"{row['area_m2']:.1f}"))
            self.results_table.setItem(r, 2, QTableWidgetItem(str(row["num_plantas"])))
            densidad = row["densidad_m2"]
            self.results_table.setItem(r, 3, QTableWidgetItem(f"{densidad:.6f}" if densidad == densidad else "-"))
            self.results_table.setItem(r, 4, QTableWidgetItem(str(row["densidad_clase"])))

    # ---------------------------------------------------------- Pintar mapa

    def _apply_categorized_renderer(self, layer, column: str, color_map: dict) -> None:
        categories = []
        for label, color in color_map.items():
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(color)
            categories.append(QgsRendererCategory(label, symbol, label))

        renderer = QgsCategorizedSymbolRenderer(column, categories)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    def on_paint_clicked(self) -> None:
        if self._last_result is None or self._analysis_layer is None:
            QMessageBox.warning(self, "Sin resultados", "Primero ejecuta 'Detectar y Contar Plantas'.")
            return

        self._apply_categorized_renderer(self._analysis_layer, "densidad_clase", _DENSITY_COLORS)
        self._log("Parcelas coloreadas por 'densidad_clase' (Alta=verde, Media=amarillo, Baja=rojo).")

    # ------------------------------------------------ Índice de vegetación

    def on_vegetation_clicked(self) -> None:
        raster_layer = self.veg_raster_combo.currentLayer()
        parcels_layer = self.parcels_combo.currentLayer()

        if raster_layer is None or parcels_layer is None:
            QMessageBox.warning(
                self, "Faltan capas",
                "Selecciona un ráster de índice (NDVI/NDRE) y una capa de parcelas."
            )
            return

        raster_path = layer_source_path(raster_layer)
        if not raster_path:
            QMessageBox.warning(
                self, "Ráster no válido",
                "No se pudo resolver la ruta del archivo del ráster de índice "
                "(debe ser una capa respaldada por un archivo, ej. GeoTIFF)."
            )
            return

        try:
            from .qgis_bridge import qgs_vector_layer_to_geodataframe
            parcelas_gdf = qgs_vector_layer_to_geodataframe(parcels_layer)
        except Exception as e:
            QMessageBox.critical(self, "Error al leer parcelas", str(e))
            return

        prefix = self.veg_index_combo.currentText()
        params = {"raster_path": raster_path, "parcelas_gdf": parcelas_gdf, "prefix": prefix}

        self.veg_calc_btn.setEnabled(False)
        self._log(f"Calculando {prefix.upper()} por parcela...")

        self._veg_task = VegetationIndexTask(params)
        self._veg_task.taskCompleted.connect(self._on_vegetation_completed)
        self._veg_task.taskTerminated.connect(self._on_vegetation_terminated)
        QgsApplication.taskManager().addTask(self._veg_task)

    def _on_vegetation_completed(self) -> None:
        self.veg_calc_btn.setEnabled(True)
        result = self._veg_task.result
        self._veg_task = None

        self._last_vegetation_result = result
        prefix = result["prefix"]
        parcelas_gdf = result["parcelas_gdf"]

        n_con_datos = parcelas_gdf[f"{prefix}_mean"].notna().sum()
        self._log(f"{prefix.upper()} calculado: {n_con_datos}/{len(parcelas_gdf)} parcela(s) con datos.")

        try:
            self._vegetation_layer = geodataframe_to_qgs_layer(parcelas_gdf, f"Parcelas - {prefix.upper()}")
            self._log(f"Capa 'Parcelas - {prefix.upper()}' añadida al proyecto.")
            self.veg_paint_btn.setEnabled(True)
        except Exception as e:
            self._log(f"No se pudo crear la capa de resultado: {e}")
            self._vegetation_layer = None

    def _on_vegetation_terminated(self) -> None:
        self.veg_calc_btn.setEnabled(True)
        error = self._veg_task.error if self._veg_task is not None else None
        self._veg_task = None

        if error is not None:
            self._log(f"Error: {error}")
            QMessageBox.critical(self, "Error al calcular el índice de vegetación", str(error))
        else:
            self._log("El cálculo del índice de vegetación terminó de forma inesperada.")

    def on_paint_vegetation_clicked(self) -> None:
        if self._last_vegetation_result is None or self._vegetation_layer is None:
            QMessageBox.warning(self, "Sin resultados", "Primero ejecuta 'Calcular Índice de Vegetación'.")
            return

        prefix = self._last_vegetation_result["prefix"]
        self._apply_categorized_renderer(self._vegetation_layer, f"{prefix}_clase", _VEGETATION_COLORS)
        self._log(
            f"Parcelas coloreadas por vigor vegetal ({prefix}_clase): "
            "Vigorosa=verde, Moderada=verde claro, Estresada=naranja, Suelo/Agua=marrón."
        )

    # --------------------------------------------------------------- Reporte

    def on_report_clicked(self) -> None:
        if self._last_result is None:
            QMessageBox.warning(self, "Sin resultados", "Primero ejecuta 'Detectar y Contar Plantas'.")
            return

        from qgis.PyQt.QtWidgets import QFileDialog

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar reporte", "conteo_por_parcela.csv", "CSV (*.csv)"
        )
        if not output_path:
            return

        try:
            from core.exports.export_manager import ExportManager
            ExportManager.export_tabular_data(
                self._last_result["counts"], output_path, columns=["Parcela", "Num_Plantas"]
            )
            self._log(f"Reporte exportado a: {output_path}")
            self.iface.messageBar().pushMessage(
                "Agricultura de Precisión", f"Reporte generado: {output_path}", level=Qgis.MessageLevel.Success
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))
