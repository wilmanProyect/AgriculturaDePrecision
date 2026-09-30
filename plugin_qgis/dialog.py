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
    QgsProject,
    QgsVectorLayer,
)
from qgis.gui import QgsFileWidget, QgsMapLayerComboBox
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QAbstractItemView,
)

from .qgis_bridge import geodataframe_to_qgs_layer, layer_source_path


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

# Colores por clase de infestación de malezas: Baja -> verde, Alta -> rojo
_INFESTATION_COLORS = {
    "Baja": QColor(26, 152, 80),
    "Media": QColor(255, 217, 47),
    "Alta": QColor(215, 48, 39),
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
        self._row_task = None
        self._row_lines_defaults = {}
        self._weed_task = None
        self._last_weed_result = None
        self._weed_layer = None
        self._weed_defaults = {}

        self.setWindowTitle("Agricultura de Precisión")
        self.resize(680, 720)
        self._area_rows = []

        self._build_ui()
        self._load_defaults()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Analizador agrícola")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        intro = QLabel("Selecciona tus polígonos y elige una herramienta.")
        layout.addWidget(intro)
        self.parcels_combo = QgsMapLayerComboBox()
        self.parcels_combo.setFilters(QgsMapLayerProxyModel.Filter.PolygonLayer)
        form = QFormLayout()
        form.addRow("Capa de polígonos:", self.parcels_combo)
        layout.addLayout(form)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        area_page = QWidget()
        area_layout = QVBoxLayout(area_page)
        hint = QLabel("Calcula la superficie sin necesitar una imagen ni un modelo.")
        hint.setWordWrap(True)
        area_layout.addWidget(hint)
        self.selected_only = QCheckBox("Medir solo los polígonos seleccionados en el mapa")
        area_layout.addWidget(self.selected_only)
        self.detect_btn = QPushButton("Calcular área")
        self.detect_btn.clicked.connect(self.on_area_clicked)
        area_layout.addWidget(self.detect_btn)
        self.area_table = QTableWidget(0, 3)
        self.area_table.setHorizontalHeaderLabels(["ID del elemento", "Área (m²)", "Área (ha)"])
        self.area_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.area_table.horizontalHeader().setStretchLastSection(True)
        area_layout.addWidget(self.area_table, 1)
        self.area_summary = QLabel("Todavía no hay mediciones.")
        self.area_summary.setWordWrap(True)
        area_layout.addWidget(self.area_summary)
        note = QLabel("Medición sobre el elipsoide WGS84. El total suma las superficies; los solapes se cuentan por cada polígono.")
        note.setWordWrap(True)
        area_layout.addWidget(note)
        self.area_export_btn = QPushButton("Exportar áreas a CSV")
        self.area_export_btn.setEnabled(False)
        self.area_export_btn.clicked.connect(self.on_area_export_clicked)
        area_layout.addWidget(self.area_export_btn)
        tabs.addTab(area_page, "Área de polígonos")

        analysis_page = QWidget()
        analysis_layout = QVBoxLayout(analysis_page)
        self.raster_combo = QgsMapLayerComboBox()
        self.raster_combo.setFilters(QgsMapLayerProxyModel.Filter.RasterLayer)
        image_form = QFormLayout()
        image_form.addRow("Ortomosaico RGB:", self.raster_combo)
        analysis_layout.addLayout(image_form)
        self.rows_rgb_btn = QPushButton("Identificar líneas de siembra y posibles fallas")
        self.rows_rgb_btn.clicked.connect(self.on_rows_rgb_clicked)
        analysis_layout.addWidget(self.rows_rgb_btn)
        row_hint = QLabel("Las líneas y posibles fallas se estiman desde la imagen. Revisa el resultado en el mapa.")
        row_hint.setWordWrap(True)
        analysis_layout.addWidget(row_hint)
        veg_group = QGroupBox("Vegetación en el ortomosaico")
        vform = QFormLayout(veg_group)
        self.veg_index_combo = QComboBox()
        self.veg_index_combo.addItems(["exg", "vari"])
        vform.addRow("Índice RGB:", self.veg_index_combo)
        self.veg_index_hint = QLabel()
        self.veg_index_hint.setWordWrap(True)
        vform.addRow(self.veg_index_hint)
        self.veg_index_combo.currentTextChanged.connect(self._update_veg_index_hint)
        self._update_veg_index_hint("exg")
        self.veg_calc_btn = QPushButton("Calcular índice de vegetación")
        self.veg_calc_btn.clicked.connect(self.on_vegetation_clicked)
        vform.addRow(self.veg_calc_btn)
        self.veg_paint_btn = QPushButton("Colorear polígonos por vegetación")
        self.veg_paint_btn.clicked.connect(self.on_paint_vegetation_clicked)
        self.veg_paint_btn.setEnabled(False)
        vform.addRow(self.veg_paint_btn)
        analysis_layout.addWidget(veg_group)
        analysis_layout.addStretch()
        tabs.addTab(analysis_page, "Análisis del ortomosaico")

        # Keep legacy controls owned and hidden so existing callbacks remain compatible.
        self._hidden_controls = QWidget(self)
        self._hidden_controls.hide()
        for name in ("paint_btn", "report_btn", "weed_detect_btn", "weed_paint_btn", "weed_report_btn", "rows_btn"):
            setattr(self, name, QPushButton(self._hidden_controls))
        for name in ("weights_widget", "weed_weights_widget"):
            setattr(self, name, QgsFileWidget(self._hidden_controls))
        self.device_combo = QComboBox(self._hidden_controls)
        self.device_combo.addItems(["cpu", "cuda"])
        for name, value in (("conf_spin", .25), ("iou_spin", .45), ("overlap_spin", .2), ("weed_conf_spin", .25)):
            control = QDoubleSpinBox(self._hidden_controls)
            control.setRange(0, 1)
            control.setValue(value)
            setattr(self, name, control)
        self.tile_spin = QSpinBox(self._hidden_controls)
        self.tile_spin.setRange(128, 4096)
        self.tile_spin.setValue(1024)
        self.results_table = QTableWidget(0, 5, self._hidden_controls)
        self.weed_results_table = QTableWidget(0, 6, self._hidden_controls)
        self.veg_raster_combo = self.raster_combo

        status = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status.addWidget(self.progress_bar)
        self.cancel_btn = QPushButton("Cancelar análisis")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.on_cancel_clicked)
        status.addWidget(self.cancel_btn)
        layout.addLayout(status)
        details = QGroupBox("Detalles de la operación")
        details.setCheckable(True)
        details.setChecked(False)
        details_layout = QVBoxLayout(details)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(500)
        self.log_output.setMaximumHeight(110)
        self.log_output.hide()
        details.toggled.connect(self.log_output.setVisible)
        details_layout.addWidget(self.log_output)
        layout.addWidget(details)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.button(QDialogButtonBox.StandardButton.Close).setText("Cerrar")
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.parcels_combo.layerChanged.connect(self._clear_area_results)
        self.selected_only.toggled.connect(self._clear_area_results)

    def _clear_area_results(self, *_):
        self._area_rows = []
        self.area_table.setRowCount(0)
        self.area_summary.setText("Pulsa Calcular área para medir los polígonos actuales.")
        self.area_export_btn.setEnabled(False)

    def on_area_clicked(self):
        from .area import measure_polygon_areas
        self._clear_area_results()
        try:
            layer = self.parcels_combo.currentLayer()
            rows = measure_polygon_areas(layer, self.selected_only.isChecked())
        except Exception as exc:
            QMessageBox.warning(self, "No se puede calcular el área", str(exc))
            return
        self._area_rows = rows
        self.area_table.setRowCount(len(rows))
        for index, (fid, m2, ha) in enumerate(rows):
            for column, value in enumerate((str(fid), f"{m2:,.2f}", f"{ha:,.4f}")):
                self.area_table.setItem(index, column, QTableWidgetItem(value))
        total = sum(row[1] for row in rows)
        self.area_summary.setText(f"{layer.name()} · {len(rows)} polígono(s) · Total: {total:,.2f} m² / {total / 10000:,.4f} ha")
        self.area_export_btn.setEnabled(True)
        self._log(self.area_summary.text())

    def on_area_export_clicked(self):
        if not self._area_rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar áreas", "areas_poligonos.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ID_elemento", "Area_m2", "Area_ha"])
                writer.writerows(self._area_rows)
            self._log(f"Áreas exportadas: {path}")
            self.iface.messageBar().pushMessage("Áreas", "CSV guardado correctamente", level=Qgis.MessageLevel.Success)
        except Exception as exc:
            QMessageBox.critical(self, "No se pudo exportar", str(exc))

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

            row_lines_cfg = ai_cfg.get("row_lines", {})
            row_weights = row_lines_cfg.get("weights_path")
            if row_weights:
                candidate = row_weights if os.path.isabs(row_weights) else os.path.join(project_root, row_weights)
                if os.path.exists(candidate):
                    self._row_lines_defaults["weights_path"] = candidate
            if "conf_threshold" in row_lines_cfg:
                self._row_lines_defaults["conf_threshold"] = float(row_lines_cfg["conf_threshold"])

            weed_cfg = ai_cfg.get("weed_detection", {})
            weed_weights = weed_cfg.get("weights_path")
            if weed_weights:
                candidate = weed_weights if os.path.isabs(weed_weights) else os.path.join(project_root, weed_weights)
                if os.path.exists(candidate):
                    self.weed_weights_widget.setFilePath(candidate)
            self.weed_conf_spin.setValue(float(weed_cfg.get("conf_threshold", 0.25)))
        except Exception as e:
            self._log(f"No se pudo cargar config/default.yaml ({e}). Usando valores por defecto.")

    def _log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def _update_veg_index_hint(self, index_type: str) -> None:
        if index_type in ("exg", "vari"):
            self.veg_index_hint.setText(
                "Se calcula al vuelo desde las bandas R/G/B: selecciona el propio "
                "ortomosaico RGB (no necesitas un ráster de índice aparte)."
            )
        else:
            self.veg_index_hint.setText(
                "Requiere un ráster de índice ya generado a partir de un vuelo con "
                "sensor NIR/RedEdge (ej. exportado por el software de fotogrametría)."
            )

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

        from .worker import DetectionTask
        self._task = DetectionTask(params)
        self._task.taskCompleted.connect(self._on_task_completed)
        self._task.taskTerminated.connect(self._on_task_terminated)
        self._task.progressChanged.connect(lambda p: self.progress_bar.setValue(int(p)))
        QgsApplication.taskManager().addTask(self._task)

    def _set_running(self, running: bool) -> None:
        self.detect_btn.setEnabled(not running)
        self.rows_btn.setEnabled(not running and self._veg_task is None)
        self.rows_rgb_btn.setEnabled(not running and self._veg_task is None)
        self.weed_detect_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.progress_bar.setValue(0)

    def on_cancel_clicked(self) -> None:
        if self._veg_task is not None:
            self._veg_task.cancel()
            self._log("Cancelando análisis de vegetación...")
            return
        if self._row_task is not None:
            self._row_task.cancel()
            self._log("Cancelando análisis de surcos...")
            return
        if self._weed_task is not None:
            self._weed_task.cancel()
            self._log("Cancelando detección de malezas...")
            return
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
        # Semitransparente: son polígonos duplicados de la capa de parcelas de entrada,
        # dibujados encima de ella; a opacidad completa tapan el ortomosaico por debajo.
        layer.setOpacity(0.55)
        layer.triggerRepaint()

    def on_paint_clicked(self) -> None:
        if self._last_result is None or self._analysis_layer is None:
            QMessageBox.warning(self, "Sin resultados", "Primero ejecuta 'Detectar y Contar Plantas'.")
            return

        self._apply_categorized_renderer(self._analysis_layer, "densidad_clase", _DENSITY_COLORS)
        self._log("Parcelas coloreadas por 'densidad_clase' (Alta=verde, Media=amarillo, Baja=rojo).")

    # -------------------------------------------------------- Malezas

    def on_weed_detect_clicked(self) -> None:
        if self._task is not None or self._veg_task is not None or self._row_task is not None or self._weed_task is not None:
            return

        raster_layer = self.raster_combo.currentLayer()
        parcels_layer = self.parcels_combo.currentLayer()
        if raster_layer is None or parcels_layer is None:
            QMessageBox.warning(self, "Faltan capas", "Selecciona un ortomosaico y una capa de parcelas.")
            return

        weights_path = self.weed_weights_widget.filePath()
        if not weights_path or not os.path.exists(weights_path):
            QMessageBox.warning(
                self, "Modelo no encontrado",
                "Selecciona un archivo de pesos YOLO (.pt) entrenado para detectar malezas."
            )
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
            "conf_threshold": self.weed_conf_spin.value(),
            "iou_threshold": self.iou_spin.value(),
            "tile_size": self.tile_spin.value(),
            "overlap": self.overlap_spin.value(),
        }

        self._set_weed_running(True)
        self._log("Iniciando detección de malezas...")

        from .worker import WeedAnalysisTask
        self._weed_task = WeedAnalysisTask(params)
        self._weed_task.taskCompleted.connect(self._on_weed_completed)
        self._weed_task.taskTerminated.connect(self._on_weed_terminated)
        self._weed_task.progressChanged.connect(lambda p: self.progress_bar.setValue(int(p)))
        QgsApplication.taskManager().addTask(self._weed_task)

    def _set_weed_running(self, running: bool) -> None:
        self.weed_detect_btn.setEnabled(not running)
        self.detect_btn.setEnabled(not running)
        self.rows_btn.setEnabled(not running and self._veg_task is None)
        self.rows_rgb_btn.setEnabled(not running and self._veg_task is None)
        self.veg_calc_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.progress_bar.setValue(0)

    def _on_weed_completed(self) -> None:
        self._set_weed_running(False)
        result = self._weed_task.result
        was_canceled = self._weed_task.isCanceled()
        self._weed_task = None

        self._last_weed_result = result
        self.progress_bar.setValue(100)
        n_malezas = len(result["malezas_poly_gdf"])
        n_cultivo = len(result["cultivo_poly_gdf"])
        if was_canceled:
            self._log(
                f"Detección de malezas cancelada. Resultados parciales: {n_malezas} malezas "
                f"y {n_cultivo} plantas de cultivo detectadas antes de cancelar."
            )
        elif result["has_crop_class"]:
            self._log(f"Detección de malezas finalizada: {n_malezas} malezas y {n_cultivo} plantas de cultivo.")
        else:
            self._log(
                f"Detección de malezas finalizada: {n_malezas} malezas detectadas "
                "(el modelo no distingue una clase de cultivo aparte; la infestación se clasificó por densidad)."
            )

        self._populate_weed_results_table(result["parcelas_gdf"])

        try:
            malezas_layer = geodataframe_to_qgs_layer(result["malezas_poly_gdf"], "Malezas detectadas")
            self._style_box_layer(malezas_layer, QColor(255, 37, 37))
            if n_cultivo > 0:
                cultivo_layer = geodataframe_to_qgs_layer(result["cultivo_poly_gdf"], "Cultivo detectado")
                self._style_box_layer(cultivo_layer, QColor(26, 152, 80))
            self._weed_layer = geodataframe_to_qgs_layer(result["parcelas_gdf"], "Parcelas - malezas")
            self._log("Capa 'Malezas detectadas' (polígonos alrededor de cada maleza) añadida al proyecto.")
        except Exception as e:
            self._log(f"No se pudieron crear las capas de resultado: {e}")
            self._weed_layer = None

        self.weed_paint_btn.setEnabled(self._weed_layer is not None)
        self.weed_report_btn.setEnabled(True)

    def _style_box_layer(self, layer, color: QColor) -> None:
        """Relleno apenas visible y borde sólido: marca el recuadro detectado sin tapar
        el ortomosaico por debajo."""
        symbol = layer.renderer().symbol()
        fill_color = QColor(color)
        fill_color.setAlpha(40)
        symbol.setColor(fill_color)
        symbol_layer = symbol.symbolLayer(0)
        symbol_layer.setStrokeColor(color)
        symbol_layer.setStrokeWidth(0.5)
        layer.triggerRepaint()

    def _on_weed_terminated(self) -> None:
        error = self._weed_task.error if self._weed_task is not None else None
        self._set_weed_running(False)
        self._weed_task = None

        if error is not None:
            self._log(f"Error: {error}")
            QMessageBox.critical(self, "Error durante la detección de malezas", str(error))
        else:
            self._log("La tarea de detección de malezas terminó de forma inesperada.")

    def _populate_weed_results_table(self, parcelas_gdf) -> None:
        self.weed_results_table.setRowCount(0)
        id_col = self._last_weed_result["id_col"]
        has_crop_class = self._last_weed_result["has_crop_class"]
        for _, row in parcelas_gdf.iterrows():
            r = self.weed_results_table.rowCount()
            self.weed_results_table.insertRow(r)
            self.weed_results_table.setItem(r, 0, QTableWidgetItem(str(row[id_col])))
            self.weed_results_table.setItem(r, 1, QTableWidgetItem(f"{row['area_m2']:.1f}"))
            self.weed_results_table.setItem(r, 2, QTableWidgetItem(str(row["num_malezas"])))
            self.weed_results_table.setItem(r, 3, QTableWidgetItem(str(row["num_cultivo"])))
            cobertura = row["cobertura_malezas"]
            if cobertura != cobertura:  # NaN
                cobertura_text = "-"
            elif has_crop_class:
                cobertura_text = f"{cobertura * 100:.1f}%"
            else:
                cobertura_text = f"{cobertura:.4f} pl/m²"
            self.weed_results_table.setItem(r, 4, QTableWidgetItem(cobertura_text))
            self.weed_results_table.setItem(r, 5, QTableWidgetItem(str(row["infestacion_clase"])))

    def on_paint_weed_clicked(self) -> None:
        if self._last_weed_result is None or self._weed_layer is None:
            QMessageBox.warning(self, "Sin resultados", "Primero ejecuta 'Detectar Malezas'.")
            return

        self._apply_categorized_renderer(self._weed_layer, "infestacion_clase", _INFESTATION_COLORS)
        self._log("Parcelas coloreadas por 'infestacion_clase' (Baja=verde, Media=amarillo, Alta=rojo).")

    def on_weed_report_clicked(self) -> None:
        if self._last_weed_result is None:
            QMessageBox.warning(self, "Sin resultados", "Primero ejecuta 'Detectar Malezas'.")
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar reporte de malezas", "malezas_por_parcela.csv", "CSV (*.csv)"
        )
        if not output_path:
            return

        try:
            id_col = self._last_weed_result["id_col"]
            parcelas_gdf = self._last_weed_result["parcelas_gdf"]
            columns = [id_col, "area_m2", "num_malezas", "num_cultivo", "cobertura_malezas", "infestacion_clase"]
            df = parcelas_gdf[columns].rename(columns={
                id_col: "Parcela",
                "area_m2": "Area_m2",
                "num_malezas": "Num_Malezas",
                "num_cultivo": "Num_Cultivo",
                "cobertura_malezas": "Cobertura_Malezas",
                "infestacion_clase": "Infestacion",
            })
            df.to_csv(output_path, index=False, encoding="utf-8")
            self._log(f"Reporte de malezas exportado a: {output_path}")
            self.iface.messageBar().pushMessage(
                "Agricultura de Precisión", f"Reporte de malezas generado: {output_path}", level=Qgis.MessageLevel.Success
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))

    # ------------------------------------------------ Índice de vegetación

    def on_rows_clicked(self) -> None:
        self._run_row_analysis(include_yolo=True)

    def on_rows_rgb_clicked(self) -> None:
        self._run_row_analysis(include_yolo=False)

    def _run_row_analysis(self, include_yolo: bool) -> None:
        if self._task is not None or self._veg_task is not None or self._row_task is not None or self._weed_task is not None:
            return
        raster = self.raster_combo.currentLayer()
        parcels = self.parcels_combo.currentLayer()
        path = layer_source_path(raster)
        if not path or parcels is None:
            QMessageBox.warning(self, "Faltan entradas", "Selecciona un ortomosaico RGB de archivo y una capa de parcelas.")
            return
        try:
            from .qgis_bridge import qgs_vector_layer_to_geodataframe
            from core.analysis.row_detection import RowOptions
            gdf = qgs_vector_layer_to_geodataframe(parcels)
            if gdf.empty or gdf.crs is None:
                raise ValueError("Las parcelas deben tener geometría y CRS definidos.")
            # Reproyectar a un CRS métrico ACÁ, en el hilo principal: calcular
            # estimate_utm_crs()/to_crs() por primera vez desde el hilo en segundo
            # plano de la tarea puede colgar QGIS con un crash nativo de PROJ.
            row_crs = gdf.estimate_utm_crs()
            if row_crs is None:
                raise ValueError("No se pudo determinar un CRS métrico para las parcelas.")
            gdf = gdf.to_crs(row_crs)
        except Exception as exc:
            QMessageBox.warning(self, "No se puede analizar", str(exc))
            return

        settings = QDialog(self)
        settings.setWindowTitle("Líneas de siembra y posibles fallas")
        form = QFormLayout(settings)
        if include_yolo:
            hint_text = ("Surcos en celeste y tramos no sembrados en rojo. Sin modelo YOLO usa una "
                        "heurística RGB (vegetación); con modelo YOLO (segmentación o pose), cada "
                        "línea es una instancia que localiza el modelo, y los huecos entre "
                        "detecciones de una misma fila también se reportan como posible falla. En "
                        "ambos casos revisa las cabeceras y los resultados: no confirma plantas "
                        "faltantes ni surcos completamente omitidos.")
        else:
            hint_text = ("Surcos en celeste y tramos no sembrados en rojo. Heurística RGB (ExG + "
                        "seguimiento de crestas), no requiere ningún modelo entrenado. Revisa las "
                        "cabeceras y los resultados: no confirma plantas faltantes ni surcos "
                        "completamente omitidos.")
        hint = QLabel(hint_text)
        hint.setWordWrap(True)
        form.addRow(hint)

        weights_widget = None
        conf_spin = None
        if include_yolo:
            weights_widget = QgsFileWidget()
            weights_widget.setFilter("Modelos YOLO (*.pt)")
            row_lines_defaults = self._row_lines_defaults
            if row_lines_defaults.get("weights_path"):
                weights_widget.setFilePath(row_lines_defaults["weights_path"])
            form.addRow("Modelo YOLO (.pt, segmentación o pose) — opcional:", weights_widget)

            conf_spin = QDoubleSpinBox()
            conf_spin.setRange(0.01, 1.0)
            conf_spin.setSingleStep(0.05)
            conf_spin.setValue(row_lines_defaults.get("conf_threshold", 0.25))
            form.addRow("Confianza mínima (solo YOLO):", conf_spin)

        resolution_auto = QCheckBox("Detectar automáticamente (recomendado)")
        resolution_auto.setChecked(True)
        resolution_spin = QDoubleSpinBox()
        resolution_spin.setDecimals(3)
        resolution_spin.setRange(.01, .5)
        resolution_spin.setSingleStep(.01)
        resolution_spin.setValue(.05)
        resolution_spin.setEnabled(False)
        resolution_auto.toggled.connect(lambda checked: resolution_spin.setEnabled(not checked))
        resolution_row = QHBoxLayout()
        resolution_row.addWidget(resolution_auto)
        resolution_row.addWidget(resolution_spin)
        resolution_hint = QLabel("Usa el detalle real del ortomosaico sin superar el límite de memoria por "
                                 "parcela; desmarcá para fijar un valor manual.")
        resolution_hint.setWordWrap(True)
        form.addRow("Resolución de análisis (m/píxel):", resolution_row)
        form.addRow("", resolution_hint)

        controls = {}
        for key, label, minimum, maximum, default, step in [
            ('spacing_m', 'Separación aproximada entre surcos (m):', .15, 10., .45, .05),
            ('min_gap_m', 'Longitud mínima de falla (m):', .1, 100., 1., .1),
            ('vegetation_threshold', 'Umbral de vegetación (mayor = más fallas) — solo heurística RGB:', -.2, .8, .065, .005),
        ]:
            control = QDoubleSpinBox()
            control.setDecimals(3)
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            control.setValue(default)
            form.addRow(label, control)
            controls[key] = control
        output = QLineEdit(os.path.join(os.path.expanduser('~'), 'Documents', 'Agroptima', 'AnalisisSiembra'))
        browse = QPushButton("Elegir carpeta")
        def choose_folder():
            folder = QFileDialog.getExistingDirectory(settings, "Guardar resultados", output.text())
            if folder:
                output.setText(folder)
        browse.clicked.connect(choose_folder)
        form.addRow("Carpeta de resultados:", output)
        form.addRow(browse)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(settings.accept)
        buttons.rejected.connect(settings.reject)
        form.addRow(buttons)
        if not settings.exec():
            return
        weights_path = ""
        conf_value = 0.25
        if include_yolo:
            weights_path = weights_widget.filePath().strip()
            if weights_path and not os.path.exists(weights_path):
                QMessageBox.warning(self, "Modelo no encontrado", "El archivo de pesos YOLO-seg indicado no existe.")
                return
            conf_value = conf_spin.value()
        try:
            resolution_m = None if resolution_auto.isChecked() else resolution_spin.value()
            options = RowOptions(resolution_m=resolution_m,
                                  **{key: value.value() for key, value in controls.items()})
            options.validate()
            if not output.text().strip():
                raise ValueError("Indica una carpeta para guardar resultados.")
        except ValueError as exc:
            QMessageBox.warning(self, "Parámetros inválidos", str(exc))
            return
        from .worker import RowAnalysisTask
        self._row_task = RowAnalysisTask(dict(raster_path=path, parcelas_gdf=gdf, options=options,
                                              output_dir=output.text().strip(), weights_path=weights_path,
                                              conf_threshold=conf_value, crs=row_crs))
        self.rows_btn.setEnabled(False)
        self.rows_rgb_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)
        self.veg_calc_btn.setEnabled(False)
        self.weed_detect_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self._row_task.progressChanged.connect(lambda value: self.progress_bar.setValue(int(value)))
        self._row_task.taskCompleted.connect(self._on_rows_completed)
        self._row_task.taskTerminated.connect(self._on_rows_terminated)
        if weights_path:
            self._log("Segmentando líneas de siembra con el modelo YOLO indicado...")
        else:
            self._log("Analizando surcos y cabeceras (heurística RGB); los tramos rojos serán posibles fallas.")
        QgsApplication.taskManager().addTask(self._row_task)

    def _finish_rows(self):
        self._row_task = None
        self.rows_btn.setEnabled(True)
        self.rows_rgb_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.veg_calc_btn.setEnabled(True)
        self.weed_detect_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_rows_completed(self):
        result = self._row_task.result
        self._finish_rows()
        suffix = "YOLO" if result.get('model_weights') else "RGB"
        try:
            for key, name, color, width in [
                ('rows_gdf', f'Líneas de siembra - {suffix}', '#00dce8', .18),
                ('gaps_gdf', 'Posibles fallas de siembra', '#ff2525', .55),
            ]:
                if key not in result['paths']:
                    continue
                layer = QgsVectorLayer(result['paths'][key], name, 'ogr')
                if not layer.isValid():
                    raise ValueError(f"No se pudo cargar {name}.")
                symbol = layer.renderer().symbol()
                symbol.setColor(QColor(color))
                symbol.setWidth(width)
                metadata = layer.metadata()
                metadata.setAbstract(result['note'])
                layer.setMetadata(metadata)
                QgsProject.instance().addMapLayer(layer)
                layer.saveNamedStyle(os.path.splitext(result['paths'][key])[0] + '.qml')
                layer.triggerRepaint()
            self.iface.mapCanvas().refresh()
            self.progress_bar.setValue(100)
            resolution_used = result.get('options', {}).get('resolution_m')
            resolution_note = f" (resolución usada: {resolution_used:.3f} m/px)" if resolution_used else ""
            self._log(
                f"Resultado: {result['rows']} línea(s) sembrada(s) ({result['row_length_m']:.1f} m) y "
                f"{result['gaps']} posible(s) falla(s) ({result['gap_length_m']:.1f} m){resolution_note}. "
                f"Guardado en {result['folder']}"
            )
            for warning in result['warnings']:
                self._log(warning)
        except Exception as exc:
            QMessageBox.critical(self, "Error al cargar resultados", f"{exc}\nArchivos guardados en {result['folder']}")

    def _on_rows_terminated(self):
        error = self._row_task.error if self._row_task else None
        self._finish_rows()
        self._log(f"Error al analizar surcos: {error}" if error else "Análisis de surcos cancelado.")
        if error:
            QMessageBox.critical(self, "Análisis de surcos", str(error))

    def on_vegetation_clicked(self) -> None:
        if any(task is not None for task in (self._task, self._veg_task, self._row_task, self._weed_task)):
            return
        raster_layer = self.veg_raster_combo.currentLayer()
        parcels_layer = self.parcels_combo.currentLayer()

        if raster_layer is None or parcels_layer is None:
            QMessageBox.warning(
                self, "Faltan capas",
                "Selecciona un ortomosaico RGB y una capa de polígonos."
            )
            return

        raster_path = layer_source_path(raster_layer)
        if not raster_path:
            QMessageBox.warning(
                self, "Ráster no válido",
                "No se pudo resolver la ruta del archivo del ráster seleccionado "
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
        self.detect_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self._log(f"Calculando {prefix.upper()} por parcela...")

        from .worker import VegetationIndexTask
        self._veg_task = VegetationIndexTask(params)
        self.rows_btn.setEnabled(False)
        self.rows_rgb_btn.setEnabled(False)
        self.weed_detect_btn.setEnabled(False)
        self._veg_task.taskCompleted.connect(self._on_vegetation_completed)
        self._veg_task.taskTerminated.connect(self._on_vegetation_terminated)
        self._veg_task.progressChanged.connect(lambda p: self.progress_bar.setValue(int(p)))
        QgsApplication.taskManager().addTask(self._veg_task)

    def _on_vegetation_completed(self) -> None:
        self.veg_calc_btn.setEnabled(True)
        result = self._veg_task.result
        self._veg_task = None
        self.detect_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        self.rows_btn.setEnabled(self._task is None and self._row_task is None)
        self.rows_rgb_btn.setEnabled(self._task is None and self._row_task is None)
        self.weed_detect_btn.setEnabled(self._task is None and self._row_task is None)

        self._last_vegetation_result = result
        self.progress_bar.setValue(100)
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
        self.detect_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.rows_btn.setEnabled(self._task is None and self._row_task is None)
        self.rows_rgb_btn.setEnabled(self._task is None and self._row_task is None)
        self.weed_detect_btn.setEnabled(self._task is None and self._row_task is None)

        if error is not None:
            self._log(f"Error: {error}")
            QMessageBox.critical(self, "Error al calcular el índice de vegetación", str(error))
        else:
            self._log("Cálculo del índice de vegetación cancelado o interrumpido.")

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
