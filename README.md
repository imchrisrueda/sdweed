# Generación de Imágenes Sintéticas Agrícolas con Stable Diffusion + LoRA

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-Research-green.svg)](LICENSE)

> **🚀 Inicio Rápido**: Ver [QUICKSTART.md](QUICKSTART.md) para comenzar en 3 pasos.

## Descripción del Proyecto

Sistema de inferencia para la generación de imágenes sintéticas fotorrealistas de cultivos agrícolas desde perspectiva aérea, utilizando Stable Diffusion v1.5 con adaptación LoRA especializada. El proyecto aborda la necesidad de aumentación de datos visuales para entrenamiento de modelos de visión por computador en agricultura de precisión.

### Problema Abordado

Generación de imágenes sintéticas fotorrealistas, estructuralmente coherentes y semánticamente controlables de cultivos objetivo (específicamente sorghum), vistas desde un plano aproximadamente paralelo al suelo. Las imágenes preservan características morfológicas esenciales observadas en imágenes reales de campo y amplían sistemáticamente el espacio de variabilidad visual, con el fin de mejorar la generalización de modelos de visión por computador entrenados con datos aumentados.

## Arquitectura Técnica

### Modelo Base
- **Stable Diffusion v1.5** (runwayml/stable-diffusion-v1-5)
- Modelo de difusión probabilística latente
- Resolución de salida: 512×512 píxeles

### Adaptación LoRA
- **Low-Rank Adaptation** entrenada en clúster HPC
- Pesos especializados para generación de imágenes agrícolas
- Control morfológico mejorado y coherencia estructural
- Peso de adaptación configurable (rango típico: 0.1-0.6)

## Estructura del Proyecto

```
sd_infer/
├── app.py                           # Aplicación web Streamlit
├── config/                          # Configuración centralizada
│   ├── __init__.py
│   └── inference_config.py          # Parámetros del sistema
├── src/                             # Módulos principales
│   ├── __init__.py
│   ├── inference.py                 # Pipeline de generación
│   └── utils.py                     # Utilidades auxiliares
├── scripts/                         # Scripts ejecutables
│   ├── generate_images.py           # Generación principal
│   └── test_inference.py            # Validación del sistema
├── models/                          # Pesos del modelo
│   └── lora_weights/
│       ├── pytorch_lora_weights.safetensors
│       ├── config.json
│       └── README.md
├── outputs/                         # Imágenes generadas
├── docs/                            # Documentación técnica
│   ├── usage.md                     # Guía de uso
│   ├── methodology.md               # Metodología técnica
│   └── streamlit_app.md             # Guía de la aplicación web
├── requirements.txt                 # Dependencias
├── instructions.txt                 # Especificaciones originales
└── README.md                        # Este archivo
```

## Requisitos del Sistema

### Hardware
- **GPU NVIDIA**: Recomendada (CUDA compatible)
  - Reduce el tiempo de inferencia de ~5 min a ~10 s por imagen
  - VRAM mínima recomendada: 8 GB
- **CPU**: Funcional pero considerablemente más lento

### Software
- Python 3.8+
- CUDA 11.7+ (para aceleración GPU)
- Librerías especificadas en `requirements.txt`

## Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd sd_infer
```

### 2. Crear entorno virtual (recomendado)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar instalación
```bash
python3 scripts/test_inference.py
```

## Uso

### 🌐 Aplicación Web Interactiva (Recomendado)

La forma más intuitiva de utilizar el sistema es mediante la aplicación web desarrollada en Streamlit.

#### Iniciar la aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador (por defecto en `http://localhost:8501`).

#### Características de la aplicación web:

**🎨 Generación Simple**
- Interfaz intuitiva para generación de imágenes con LoRA
- Configuración de parámetros en tiempo real
- Vista previa inmediata de resultados
- Descarga directa de imágenes generadas

**⚖️ Comparación Base vs LoRA**
- Generación comparativa lado a lado
- Evaluación visual del impacto del adaptador LoRA
- Descargas independientes de cada versión

**📚 Galería e Historial**
- Registro automático de todas las generaciones
- Acceso rápido a parámetros utilizados
- Trazabilidad completa de experimentos

**ℹ️ Información del Sistema**
- Documentación integrada
- Descripción de parámetros técnicos
- Estado del hardware en tiempo real

La aplicación incluye validación automática de hardware (GPU/CPU) y proporciona feedback visual durante el proceso de generación.

---

### Generación por Línea de Comandos

Para usuarios avanzados o integración en pipelines automatizados:

#### Generación comparativa (Base vs LoRA)
```bash
python3 scripts/generate_images.py
```

Este script genera dos imágenes:
- `outputs/output_base.png`: Generación con modelo SD base
- `outputs/output_lora_w0.3.png`: Generación con adaptación LoRA

### Generación Personalizada

```python
from src import StableDiffusionInference
from config import InferenceConfig

# Inicializar sistema
inference = StableDiffusionInference(config=InferenceConfig)
inference.initialize_pipeline(load_lora=True)

# Generar imagen personalizada
image = inference.generate_image(
    prompt="aerial view, agricultural field, sorghum plant, early growth",
    seed=42,
    num_inference_steps=30,
    guidance_scale=7.5,
)

# Guardar resultado
image.save("outputs/custom_output.png")
```

## Configuración

Los parámetros del sistema se centralizan en [`config/inference_config.py`](config/inference_config.py):

### Parámetros Principales
- `MODEL_ID`: Identificador del modelo base HuggingFace
- `LORA_WEIGHTS_PATH`: Ruta a pesos LoRA entrenados
- `DEFAULT_PROMPT`: Prompt base para generación
- `LORA_ADAPTER_WEIGHTS`: Peso de influencia LoRA (default: 0.3)
- `DEFAULT_INFERENCE_PARAMS`: Parámetros de generación
  - `num_inference_steps`: 30 (pasos de denoising)
  - `guidance_scale`: 7.5 (classifier-free guidance)
  - `height` / `width`: 512 píxeles

### Modificación de Parámetros

Para ajustar el peso de influencia del LoRA:
```python
# En config/inference_config.py
LORA_ADAPTER_WEIGHTS = [0.5]  # Aumentar influencia del LoRA
```

## Validación del Sistema

### Prueba de Configuración
```bash
python3 scripts/test_inference.py
```

Este script verifica:
- Disponibilidad de CUDA
- Existencia de pesos LoRA
- Funcionalidad del pipeline
- Capacidad de escritura en directorio de salida

## Resultados

Las imágenes generadas presentan:
- **Fotorrealismo**: Textura natural del suelo y morfología vegetal coherente
- **Control semántico**: Fidelidad al prompt de entrada
- **Coherencia estructural**: Preservación de características morfológicas
- **Variabilidad sistemática**: Espacio aumentado de condiciones visuales

### Comparación Base vs LoRA

El adaptador LoRA introduce:
- Mayor coherencia en la estructura morfológica del cultivo
- Mejor alineación con características de imágenes reales de campo
- Control semántico mejorado para atributos agrícolas específicos

## Documentación Adicional

- [`docs/streamlit_app.md`](docs/streamlit_app.md): Guía completa de la aplicación web
- [`docs/usage.md`](docs/usage.md): Guía detallada de uso y ejemplos
- [`docs/methodology.md`](docs/methodology.md): Metodología técnica y fundamentos
- [`models/lora_weights/README.md`](models/lora_weights/README.md): Detalles de los pesos LoRA

## Dependencias Principales

```
diffusers==0.26.3         # Pipeline de Stable Diffusion
transformers==4.37.2      # Modelos de transformers
accelerate==0.26.1        # Optimización de carga de modelos
safetensors==0.4.2        # Formato seguro de pesos
torch                     # Framework de deep learning
Pillow==10.2.0            # Manipulación de imágenes
streamlit==1.31.0         # Framework para aplicación web
```

Ver [`requirements.txt`](requirements.txt) para lista completa.

## Contribuciones

Este proyecto es parte de un sistema de investigación en agricultura de precisión. Para contribuciones o reportes de problemas, seguir las directrices estándar de desarrollo colaborativo.

## Licencia

[Especificar licencia según requerimientos institucionales]

## Contacto

[Información de contacto del equipo de investigación]

---

**Nota**: Este sistema ha sido diseñado para fines de investigación científica en agricultura de precisión y computer vision aplicada al sector agrícola.
