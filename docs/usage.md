# Guía de Uso - Sistema de Inferencia SD + LoRA

## Introducción

Esta guía proporciona instrucciones detalladas para la utilización del sistema de generación de imágenes sintéticas agrícolas basado en Stable Diffusion con adaptación LoRA.

## Casos de Uso

### 1. Generación Comparativa Estándar

Ejecutar el script principal para generar imágenes comparativas entre el modelo base y el modelo adaptado con LoRA:

```bash
python3 scripts/generate_images.py
```

**Salidas generadas:**
- `outputs/output_base.png`: Imagen generada con SD v1.5 sin adaptación
- `outputs/output_lora_w0.3.png`: Imagen generada con adaptación LoRA (peso 0.3)

**Tiempo de ejecución estimado:**
- GPU (CUDA): ~20 segundos
- CPU: ~10 minutos

### 2. Validación del Sistema

Antes de ejecutar experimentos extensivos, validar la configuración del sistema:

```bash
python3 scripts/test_inference.py
```

Este script verifica:
- Disponibilidad de hardware de aceleración
- Integridad de archivos de pesos LoRA
- Funcionalidad del pipeline de inferencia
- Permisos de escritura en directorio de salida

### 3. Generación Programática Personalizada

#### Caso 3.1: Generación Simple con LoRA

```python
from src import StableDiffusionInference
from config import InferenceConfig

# Inicializar sistema
inference = StableDiffusionInference(config=InferenceConfig)
inference.initialize_pipeline(load_lora=True)

# Generar imagen
image = inference.generate_image(
    prompt="aerial photograph, agricultural field, sorghum plant, mature stage",
    seed=42,
)

# Guardar
image.save("outputs/sorghum_mature.png")
```

#### Caso 3.2: Exploración de Variabilidad con Múltiples Seeds

```python
from src import StableDiffusionInference
from config import InferenceConfig

inference = StableDiffusionInference(config=InferenceConfig)
inference.initialize_pipeline(load_lora=True)

prompt = "aerial view, agricultural field, sorghum plant, early growth"

# Generar variaciones con diferentes seeds
for seed in range(1000, 1010):
    image = inference.generate_image(prompt=prompt, seed=seed)
    image.save(f"outputs/variation_seed{seed}.png")
    print(f"Generada imagen con seed {seed}")
```

#### Caso 3.3: Barrido de Pesos LoRA

```python
from src import StableDiffusionInference
from config import InferenceConfig

prompt = "aerial photograph, agricultural field, sorghum plant"
seed = 1234

# Evaluar diferentes intensidades de adaptación LoRA
for weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    # Modificar temporalmente el peso de LoRA
    InferenceConfig.LORA_ADAPTER_WEIGHTS = [weight]
    
    inference = StableDiffusionInference(config=InferenceConfig)
    inference.initialize_pipeline(load_lora=True)
    
    image = inference.generate_image(prompt=prompt, seed=seed)
    image.save(f"outputs/lora_weight_{weight:.1f}.png")
    print(f"Generada imagen con peso LoRA {weight}")
```

### 4. Exploración de Prompts

#### Estructura de Prompt Recomendada

Para obtener resultados óptimos, estructurar prompts con los siguientes componentes:

```
[tipo_vista], [contexto_espacial], [características_ambientales], [especificación_cultivo], [estadío_fenológico]
```

**Ejemplos:**

```python
prompts = [
    # Variación en estadío fenológico
    "aerial drone photograph, top-down view, agricultural field, natural soil texture, realistic, sorghum plant, seedling stage",
    "aerial drone photograph, top-down view, agricultural field, natural soil texture, realistic, sorghum plant, vegetative stage",
    "aerial drone photograph, top-down view, agricultural field, natural soil texture, realistic, sorghum plant, reproductive stage",
    
    # Variación en condiciones ambientales
    "aerial photograph, agricultural field, dry soil, clear sky, sorghum plant, early growth",
    "aerial photograph, agricultural field, moist soil, overcast lighting, sorghum plant, early growth",
    
    # Variación en densidad de cultivo
    "aerial view, agricultural field, sparse plant distribution, sorghum seedlings",
    "aerial view, agricultural field, dense plant distribution, sorghum mature plants",
]

inference = StableDiffusionInference(config=InferenceConfig)
inference.initialize_pipeline(load_lora=True)

for idx, prompt in enumerate(prompts):
    image = inference.generate_image(prompt=prompt, seed=1234)
    image.save(f"outputs/prompt_variation_{idx:02d}.png")
```

### 5. Ajuste de Parámetros de Inferencia

#### Guidance Scale

El parámetro `guidance_scale` controla el nivel de adherencia al prompt:
- **Valores bajos (1-5)**: Mayor creatividad, menor fidelidad al prompt
- **Valores medios (7-9)**: Balance entre creatividad y control
- **Valores altos (10-15)**: Máxima fidelidad al prompt, riesgo de artefactos

```python
from src import StableDiffusionInference
from config import InferenceConfig

inference = StableDiffusionInference(config=InferenceConfig)
inference.initialize_pipeline(load_lora=True)

prompt = "aerial view, agricultural field, sorghum plant"

for guidance in [5.0, 7.5, 10.0, 12.5, 15.0]:
    image = inference.generate_image(
        prompt=prompt,
        seed=1234,
        guidance_scale=guidance,
    )
    image.save(f"outputs/guidance_{guidance:.1f}.png")
```

#### Número de Pasos de Inferencia

El parámetro `num_inference_steps` define el número de iteraciones de denoising:
- **Valores bajos (15-25)**: Generación rápida, menor calidad
- **Valores medios (30-50)**: Balance calidad-velocidad (recomendado)
- **Valores altos (50-100)**: Máxima calidad, convergencia refinada

```python
for steps in [20, 30, 40, 50]:
    image = inference.generate_image(
        prompt=prompt,
        seed=1234,
        num_inference_steps=steps,
    )
    image.save(f"outputs/steps_{steps}.png")
```

## Modificación de Configuración

### Cambio de Modelo Base

Para utilizar un modelo base diferente, modificar en [`config/inference_config.py`](../config/inference_config.py):

```python
MODEL_ID = "stabilityai/stable-diffusion-2-1"  # Ejemplo: SD 2.1
```

### Ajuste de Resolución

Para generar imágenes con resolución diferente:

```python
# En config/inference_config.py
DEFAULT_INFERENCE_PARAMS: Dict[str, Any] = {
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "height": 768,  # Modificado
    "width": 768,   # Modificado
}
```

**Nota**: Resoluciones superiores requieren mayor VRAM y tiempo de cómputo.

## Gestión de Memoria GPU

### Optimización para GPUs con VRAM Limitada

```python
from src import StableDiffusionInference
from config import InferenceConfig
import torch

# Habilitar optimizaciones de memoria
inference = StableDiffusionInference(config=InferenceConfig)
inference.initialize_pipeline(load_lora=True)

# Activar attention slicing (reduce uso de VRAM)
inference.pipe.enable_attention_slicing()

# Generar imagen
image = inference.generate_image(prompt="aerial view, sorghum field", seed=42)
```

### Liberación de Memoria

```python
import torch
import gc

# Después de generar imágenes
del inference
gc.collect()
torch.cuda.empty_cache()
```

## Almacenamiento con Metadatos

Para trazabilidad experimental, embeber parámetros de generación en los archivos de imagen:

```python
from src import StableDiffusionInference, save_image_with_metadata
from config import InferenceConfig

inference = StableDiffusionInference(config=InferenceConfig)
inference.initialize_pipeline(load_lora=True)

prompt = "aerial view, agricultural field, sorghum plant"
seed = 42
guidance = 7.5
steps = 30

image = inference.generate_image(
    prompt=prompt,
    seed=seed,
    num_inference_steps=steps,
    guidance_scale=guidance,
)

# Guardar con metadatos
metadata = {
    "prompt": prompt,
    "seed": seed,
    "guidance_scale": guidance,
    "num_inference_steps": steps,
    "model": InferenceConfig.MODEL_ID,
    "lora_weight": InferenceConfig.LORA_ADAPTER_WEIGHTS[0],
}

from pathlib import Path
save_image_with_metadata(
    image,
    Path("outputs/image_with_metadata.png"),
    metadata
)
```

## Solución de Problemas Comunes

### Error: CUDA out of memory

**Solución:**
1. Reducir resolución de imagen (512x512 → 384x384)
2. Habilitar `enable_attention_slicing()`
3. Reducir `num_inference_steps`
4. Ejecutar en CPU (modificar `DEVICE` en config)

### Error: LoRA weights not found

**Solución:**
Verificar que el archivo existe en la ruta especificada:
```bash
ls -lh models/lora_weights/pytorch_lora_weights.safetensors
```

### Generación muy lenta en CPU

**Esperado:** El proceso de inferencia en CPU es significativamente más lento (~10 minutos por imagen). Considerar:
1. Uso de GPU NVIDIA con CUDA
2. Reducción de `num_inference_steps` a 20-25
3. Generación batch durante períodos de baja actividad

## Referencias Técnicas

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [LoRA: Low-Rank Adaptation (Paper)](https://arxiv.org/abs/2106.09685)
- [Stable Diffusion v1.5 Model Card](https://huggingface.co/runwayml/stable-diffusion-v1-5)
