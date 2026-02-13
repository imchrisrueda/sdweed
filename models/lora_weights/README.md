# Pesos LoRA - Adaptación para Imágenes Agrícolas Aéreas

## Descripción

Este directorio contiene los pesos del adaptador LoRA (Low-Rank Adaptation) entrenado específicamente para la generación de imágenes sintéticas de cultivos agrícolas desde perspectiva aérea, con énfasis en sorghum en estadíos tempranos de crecimiento.

## Contenido

### pytorch_lora_weights.safetensors
Archivo de pesos del adaptador LoRA en formato SafeTensors. Este formato proporciona:
- Carga segura y eficiente de tensores
- Protección contra ejecución de código arbitrario
- Compatibilidad con Diffusers y transformers

**Tamaño:** ~3-10 MB (típico para adaptadores LoRA de rango bajo)

**Arquitectura objetivo:** UNet de Stable Diffusion v1.5

### config.json
Configuración del adaptador LoRA, especificando:
- Rango de la descomposición de bajo rango
- Capas del UNet modificadas
- Parámetros de inicialización

## Especificaciones Técnicas

### Modelo Base Compatible
- **Stable Diffusion v1.5** (runwayml/stable-diffusion-v1-5)
- Otros checkpoints basados en SD v1.5 son compatibles

### Arquitectura LoRA
- **Tipo:** Low-Rank Adaptation para capas de atención del UNet
- **Rango:** Consultar `config.json` para valor específico
- **Parámetros entrenables:** ~0.1-1% del modelo completo
- **Capas adaptadas:** Cross-attention y self-attention del UNet

### Entrenamiento

El adaptador LoRA fue entrenado en un clúster HPC utilizando:
- **Dataset:** Imágenes aéreas agrícolas de cultivos de sorghum
- **Perspectiva:** Vista cenital (top-down), simulando captura desde UAV
- **Resolución de entrenamiento:** 512×512 píxeles
- **Estadíos fenológicos:** Énfasis en etapas tempranas de crecimiento

**Objetivo del entrenamiento:** Mejorar la coherencia estructural y fidelidad morfológica de cultivos en imágenes generadas, preservando características observadas en imágenes reales de campo.

## Uso

### Carga Programática

```python
from diffusers import StableDiffusionPipeline

# Inicializar pipeline base
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Cargar pesos LoRA
pipe.load_lora_weights("models/lora_weights/pytorch_lora_weights.safetensors")

# Configurar peso del adaptador
pipe.set_adapters(["default"], adapter_weights=[0.3])
```

### Rango Óptimo de Pesos

El parámetro `adapter_weights` controla la intensidad de la adaptación:

| Peso | Efecto | Uso Recomendado |
|------|--------|-----------------|
| 0.1-0.2 | Adaptación sutil | Mantener generalidad del modelo base |
| 0.3-0.4 | Balance adaptación/preservación | **Recomendado para uso estándar** |
| 0.5-0.6 | Fuerte especialización | Máxima fidelidad a morfología específica |
| >0.7 | Sobreajuste | No recomendado, riesgo de artefactos |

## Evaluación de Calidad

### Comparación Base vs LoRA

El adaptador LoRA introduce las siguientes mejoras observadas:

**Coherencia estructural:**
- Mayor regularidad en patrones de distribución espacial de plantas
- Consistencia mejorada en escala y proporción de elementos vegetales

**Fidelidad morfológica:**
- Captación de características foliares específicas de sorghum
- Textura de suelo más representativa de entornos agrícolas reales

**Control semántico:**
- Respuesta mejorada a términos específicos del dominio en prompts
- Mejor diferenciación entre estadíos fenológicos mediante prompt

### Limitaciones

- **Especificidad de dominio:** Optimizado para sorghum; otros cultivos pueden requerir reentrenamiento
- **Perspectiva fija:** Entrenado para vistas cenitales; perspectivas oblicuas pueden ser subóptimas
- **Resolución:** Optimizado para 512×512; resoluciones superiores no garantizan mejora proporcional

## Compatibilidad

### Versiones de Librerías Verificadas
- `diffusers >= 0.26.0`
- `transformers >= 4.37.0`
- `safetensors >= 0.4.0`

### Formato del Archivo
- **SafeTensors:** Formato binario seguro
- **Compatibilidad:** Diffusers (HuggingFace), ComfyUI, Automatic1111 (con adaptadores)

## Origen y Trazabilidad

**Fuente:** Entrenamiento realizado en clúster HPC

**Fecha de entrenamiento:** [Especificar si está disponible]

**Versión:** [Especificar versionado si aplica]

## Licencia y Uso

**Restricciones:** Uso académico y de investigación en agricultura de precisión

**Distribución:** [Especificar política de distribución según requerimientos institucionales]

## Referencias

Para detalles sobre la metodología de entrenamiento y aplicación, consultar:
- [`docs/methodology.md`](../../docs/methodology.md): Fundamentos técnicos de LoRA y difusión
- [`docs/usage.md`](../../docs/usage.md): Guía práctica de uso del adaptador

## Contacto

Para consultas sobre los pesos LoRA, características del entrenamiento o solicitud de adaptadores específicos, contactar al equipo de investigación.

---

**Nota Técnica:** Los pesos LoRA no contienen el modelo base completo, solo las matrices de adaptación de bajo rango. Es necesario descargar el modelo base SD v1.5 separadamente desde HuggingFace Hub.
