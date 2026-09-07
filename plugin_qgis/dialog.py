# -*- coding: utf-8 -*-
"""
Diálogo principal para el plugin de Agricultura de Precisión.
"""

import os
from PyQt5 import QtWidgets, uic

# Cargar dinámicamente el archivo .ui diseñado con Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'ui', 'plugin_dialog.ui'
))

class PrecisionAgDialog(QtWidgets.QDialog, FORM_CLASS):
    """Clase del diálogo del plugin."""
    
    def __init__(self, parent=None):
        """Constructor."""
        super(PrecisionAgDialog, self).__init__(parent)
        self.setupUi(self)
        self.setup_connections()

    def setup_connections(self):
        """Configura los eventos de los widgets de la interfaz."""
        # TODO: Conectar los botones de la interfaz con sus funciones del core/ai
        pass
