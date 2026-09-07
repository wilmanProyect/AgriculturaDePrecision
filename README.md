# 🌱 AgriculturaDePrecision

## Sistema Inteligente de Conteo de Plantas y Análisis Agrícola utilizando QGIS, Python e Inteligencia Artificial

---

# Objetivo

Desarrollar un software profesional para Agricultura de Precisión capaz de:

- Detectar automáticamente plantas mediante Inteligencia Artificial.
- Contar plantas por parcela.
- Calcular áreas sembradas.
- Calcular densidad de plantas.
- Detectar espacios vacíos.
- Calcular cobertura vegetal.
- Generar mapas temáticos.
- Exportar reportes.
- Integrarse completamente con QGIS mediante un Plugin.

---

# Arquitectura General

```
                    AgriculturaDePrecision

                           │

        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼

     Motor IA          Motor GIS         Plugin QGIS

 (YOLO, SAM)      (Rasterio, GeoPandas)    (PyQGIS)
```

La lógica del sistema NO estará dentro de QGIS.

QGIS será únicamente la interfaz gráfica.

Toda la lógica se desarrollará como una biblioteca Python independiente.

---

# Tecnologías

## Lenguaje

- Python 3.12

## GIS

- QGIS LTR
- GDAL
- Rasterio
- GeoPandas
- Fiona
- PyProj
- Shapely

## Inteligencia Artificial

- PyTorch
- Ultralytics YOLO
- Segment Anything (SAM)

## Procesamiento de imágenes

- OpenCV
- NumPy
- Scikit-Image

## Ciencia de Datos

- Pandas
- Matplotlib
- Scikit-Learn

## Interfaz

- PyQt5
- Qt Designer

## Reportes

- OpenPyXL
- ReportLab
- Pandas

---

# Arquitectura del proyecto

```
AgriculturaDePrecision/

│

├── ai/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── segmentation/
│   └── utils/
│

├── core/
│   ├── gis/
│   ├── analysis/
│   ├── reports/
│   ├── exports/
│   └── geometry/
│

├── plugin_qgis/
│   ├── ui/
│   ├── icons/
│   ├── resources/
│   ├── plugin.py
│   └── dialog.py
│

├── data/
│   ├── images/
│   ├── orthomosaics/
│   ├── parcels/
│   ├── models/
│   └── outputs/
│

├── notebooks/

├── tests/

├── docs/

├── config/

├── main.py

└── README.md
```

---

# Flujo del sistema

```
Imagen Drone

↓

Ortomosaico

↓

Preprocesamiento

↓

IA

↓

Detección de plantas

↓

Georreferenciación

↓

Cruce con parcelas

↓

Conteo

↓

Análisis

↓

Mapa temático

↓

Reporte
```

---

# Fases del proyecto

---

## Fase 1

### Motor GIS

Objetivos

- Leer GeoTIFF
- Leer Shapefile
- Leer GeoPackage
- Obtener CRS
- Obtener resolución
- Obtener extensión
- Calcular área
- Calcular perímetro

---

## Fase 2

### Inteligencia Artificial

Objetivos

Entrenar un modelo YOLO para detectar:

- Plantas
- Árboles
- Espacios vacíos

Salida

Cada planta será un punto georreferenciado.

---

## Fase 3

### Conteo por parcela

Utilizar algoritmo

Point In Polygon

para determinar

```
Planta

↓

¿Dentro del polígono?

↓

Sí

↓

Contador++
```

Resultado

| Parcela | Plantas |
|----------|----------|
| A | 350 |
| B | 521 |

---

## Fase 4

### Área sembrada

Calcular

Área cubierta

Cobertura vegetal

Área libre

Área perdida

---

## Fase 5

### Densidad

```
Densidad

=

Número de plantas

/

Área sembrada
```

---

## Fase 6

### Espacios vacíos

Analizar distancia entre plantas.

Si

Distancia > distancia esperada

↓

Planta faltante

---

## Fase 7

### Colorear parcelas

Clasificar

Alta densidad

Media

Baja

Generar simbología automáticamente en QGIS.

---

## Fase 8

### Reportes

Exportar

Excel

PDF

GeoPackage

Shapefile

CSV

---

# Inteligencia Artificial

Modelo principal

YOLO

Funciones

- Detectar plantas
- Detectar árboles
- Detectar malezas

Modelo secundario

SAM

Funciones

- Segmentar cobertura vegetal
- Obtener área exacta ocupada

---

# Algoritmos

## Conteo

YOLO

↓

Bounding Boxes

↓

Centro

↓

Puntos

↓

Conteo

---

## Cruce espacial

Point In Polygon

---

## Cobertura

Segmentación

↓

Pixeles

↓

Área

---

## Densidad

```
plantas / m²
```

---

## Vacíos

KDTree

↓

Vecino más cercano

↓

Comparar distancia

---

# Plugin QGIS

El Plugin solamente será la interfaz.

Tendrá botones.

```
Abrir Imagen

Abrir Parcelas

Detectar Plantas

Contar Plantas

Calcular Área

Generar Reporte

Pintar Parcelas
```

No contendrá lógica.

Toda la lógica estará dentro de

```
core/
```

---

# Buenas prácticas

Separar

Interfaz

Procesamiento GIS

IA

Exportaciones

Configuraciones

No escribir lógica dentro del Plugin.

Todo debe implementarse mediante clases.

Cada módulo debe ser independiente.

Utilizar tipado.

Documentar funciones.

Agregar pruebas unitarias.

---

# Objetivos finales

El sistema deberá ser capaz de:

✅ Abrir ortomosaicos

✅ Detectar automáticamente plantas

✅ Contar plantas

✅ Calcular área sembrada

✅ Detectar vacíos

✅ Calcular cobertura vegetal

✅ Calcular densidad

✅ Generar mapas temáticos

✅ Exportar reportes

✅ Integrarse completamente con QGIS

---

# Futuras mejoras

- NDVI
- GNDVI
- SAVI
- Detección de enfermedades
- Conteo de frutos
- Estimación de producción
- Procesamiento por lotes
- Integración con drones DJI
- Integración con Pix4D
- Integración con Agisoft
- Integración con PostGIS
- API REST
- Aplicación Web
- Aplicación móvil

---

# Visión del proyecto

Construir una plataforma profesional de Agricultura de Precisión basada en software libre, capaz de competir con soluciones comerciales como Pix4Dfields, DroneDeploy o Agisoft Metashape en las tareas de análisis agrícola, manteniendo una arquitectura modular, extensible y reutilizable.