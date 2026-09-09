# -*- coding: utf-8 -*-
"""
QGIS Plugin para Agricultura de Precisión.
"""

import os.path

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction


class PrecisionAgPlugin:
    """Clase principal del plugin de QGIS."""

    def __init__(self, iface):
        """Constructor."""
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = self.tr(u'&Agricultura de Precisión')
        self.dlg = None

    def tr(self, message):
        """Traduce el mensaje utilizando el traductor de QGIS."""
        return QCoreApplication.translate('PrecisionAgPlugin', message)

    def initGui(self):
        """Inicializa la interfaz gráfica de usuario del plugin."""
        icon_path = os.path.join(self.plugin_dir, 'icons', 'logo.png')
        self.add_action(
            icon_path,
            text=self.tr(u'Analizador Agrícola'),
            callback=self.run,
            parent=self.iface.mainWindow()
        )

    def unload(self):
        """Elimina los elementos del plugin al desactivarlo."""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr(u'&Agricultura de Precisión'), action)
            self.iface.removeToolBarIcon(action)

    def add_action(self, icon_path, text, callback, parent):
        """Añade una acción a la toolbar y menú de QGIS."""
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        self.iface.addPluginToMenu(self.menu, action)
        self.iface.addToolBarIcon(action)
        self.actions.append(action)
        return action

    def run(self):
        """Ejecuta el diálogo del plugin."""
        from .dialog import PrecisionAgDialog

        if not self.dlg:
            self.dlg = PrecisionAgDialog(self.iface, parent=self.iface.mainWindow())
        self.dlg.show()
        self.dlg.exec()
