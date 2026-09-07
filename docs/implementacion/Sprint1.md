# Sprint 1 - Motor GIS

**Proyecto:** AgriculturaDePrecision

**Versión:** 1.0

**Estado:** En desarrollo

**Duración estimada:** 1 Sprint (1-2 semanas)

---

# Objetivo del Sprint

Desarrollar el **Motor GIS** del sistema, el cual será responsable de gestionar toda la información geoespacial utilizada por la plataforma.

Este módulo será independiente de QGIS y de la Inteligencia Artificial.

Debe ser reutilizable desde:

- Plugin QGIS
- Consola Python
- API REST (futuro)
- Aplicación Web (futuro)

---

# Objetivos

Al finalizar este Sprint el sistema deberá ser capaz de:

✅ Leer imágenes GeoTIFF

✅ Leer Shapefile

✅ Leer GeoPackage

✅ Leer KML

✅ Obtener metadatos

✅ Obtener resolución espacial

✅ Obtener CRS

✅ Obtener Bounding Box

✅ Calcular área de parcelas

✅ Calcular perímetro

✅ Validar geometrías

✅ Exportar información

---

# Alcance

Este Sprint NO incluye:

- Inteligencia Artificial
- YOLO
- Segmentación
- Plugin QGIS
- Reportes PDF
- Base de datos

---

# Arquitectura

```
                   Motor GIS

        ┌──────────────────────────┐
        │                          │
        │      GIS Engine          │
        │                          │
        └─────────────┬────────────┘
                      │
      ┌───────────────┼────────────────┐
      │               │                │
      ▼               ▼                ▼

 RasterManager   VectorManager   GeometryManager

      │               │                │

      └───────────────┼────────────────┘
                      ▼

             SpatialAnalysis
```

---

# Estructura del módulo

```
src/

    gis/

        raster/

            raster_manager.py

            raster_metadata.py

            raster_utils.py

        vector/

            vector_manager.py

            vector_metadata.py

            vector_utils.py

        geometry/

            geometry_manager.py

            geometry_validator.py

        analysis/

            spatial_analysis.py

        exports/

            export_manager.py
```

---

# Responsabilidades

## RasterManager

Responsable de:

- Abrir GeoTIFF
- Leer bandas
- Leer resolución
- Leer CRS
- Leer transformación
- Leer extensión
- Obtener tamaño
- Obtener estadísticas

No debe modificar imágenes.

---

## VectorManager

Responsable de:

- Abrir Shapefile
- Abrir GeoPackage
- Abrir KML
- Leer atributos
- Leer geometrías
- Obtener CRS
- Obtener extensión

---

## GeometryManager

Responsable de:

- Validar geometrías
- Reparar geometrías inválidas
- Calcular áreas
- Calcular perímetros
- Calcular centroides

---

## SpatialAnalysis

Responsable de:

- Intersecciones
- Contención
- Distancias
- Buffer
- Overlay

---

# Tecnologías

Python 3.12

Rasterio

GeoPandas

Shapely

Fiona

PyProj

GDAL

NumPy

Pandas

---

# Diseño Orientado a Objetos

Cada módulo deberá implementarse utilizando clases.

Ejemplo

```
RasterManager

+ open()

+ close()

+ metadata()

+ resolution()

+ extent()

+ statistics()

+ bands()
```

---

# Flujo esperado

```
GeoTIFF

↓

RasterManager

↓

Raster Metadata

↓

Spatial Analysis

↓

Resultado
```

---

# Metadatos requeridos

El sistema deberá obtener como mínimo:

Nombre

Ruta

Ancho

Alto

Número de bandas

CRS

Resolución X

Resolución Y

Transformación

Bounding Box

Tipo de dato

NoData

Tamaño del archivo

---

# Información de capas vectoriales

Nombre

Tipo

Número de entidades

CRS

Campos

Área total

Perímetro total

Bounding Box

---

# Reglas de desarrollo

Todo el código deberá utilizar:

PEP8

Type Hints

Docstrings

Logging

Manejo de excepciones

No utilizar variables globales.

No mezclar lógica GIS con lógica de IA.

---

# Logging

Crear un logger central.

Registrar:

Carga de archivos

Errores

Advertencias

Tiempo de procesamiento

---

# Excepciones

Crear excepciones personalizadas.

Ejemplo

```
RasterNotFoundError

InvalidCRSError

InvalidGeometryError

UnsupportedFormatError
```

---

# Pruebas Unitarias

Crear pruebas para:

Abrir GeoTIFF

Abrir Shape

Calcular área

Calcular perímetro

Validar geometrías

Leer CRS

---

# Datos de prueba

Crear una carpeta

```
tests/data/

ortomosaico.tif

parcelas.shp

parcelas.gpkg

parcelas.kml
```

---

# Entregables

El Sprint deberá entregar:

✔ Motor GIS completamente funcional

✔ Código documentado

✔ Tipado

✔ Tests

✔ Ejemplos de uso

✔ Documentación técnica

---

# Criterios de aceptación

El Sprint será considerado terminado cuando:

- El sistema pueda abrir correctamente un GeoTIFF.

- El sistema pueda abrir correctamente un Shapefile.

- El sistema pueda abrir un GeoPackage.

- El sistema pueda abrir un KML.

- Se puedan obtener correctamente los metadatos del raster.

- Se puedan obtener correctamente los metadatos del vector.

- Se pueda calcular el área de cualquier parcela.

- Se pueda calcular el perímetro.

- Se puedan validar geometrías.

- Todos los tests sean satisfactorios.

---

# Roadmap posterior

Una vez finalizado este Sprint se iniciará:

Sprint 2

Motor IA

donde se implementará:

- Detección de plantas

- Entrenamiento YOLO

- Segmentación

- Preparación del Dataset

---

# Definición de terminado (Definition of Done)

✔ Código revisado.

✔ Sin errores críticos.

✔ Cobertura mínima de pruebas del 80%.

✔ Documentación actualizada.

✔ Cumplimiento de PEP8.

✔ Clases desacopladas.

✔ Arquitectura modular.

✔ Preparado para integración con el Plugin QGIS.

---

# Visión

Este módulo constituye el núcleo geoespacial de la plataforma AgriculturaDePrecision. Todas las funcionalidades futuras (conteo de plantas, detección de malezas, cálculo de cobertura, análisis temporal y reportes) dependerán de este componente. Su diseño debe priorizar la reutilización, el rendimiento, la mantenibilidad y la independencia de la interfaz de usuario.