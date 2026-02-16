# Actualización del Sistema de Inferencia

**Fecha**: 16 de febrero de 2026  
**Cambios basados en**: Script HPC optimizado para generación plant-centric

## 🎯 Resumen de Cambios

El sistema de inferencia ha sido actualizado para alinearse con el código de producción del HPC, incorporando mejoras en la metodología de generación de imágenes con enfoque en plantas individuales.

---

## 📋 Cambios Principales

### 1. **Configuración (`config/inference_config.py`)**

#### Prompts Plant-Centric
- ✅ Añadidos 4 prompts especializados en plantas individuales:
  - Maize plant (maíz)
  - Sorghum plant (sorgo)  
  - Atriplex plant
  - Seedling (plántula)
- ❌ Eliminado enfoque en vistas aéreas (aerial, drone, top-down field)
- ✅ Énfasis en: `centered subject, close-up crop, soil background, natural light, sharp focus`

#### Negative Prompt
```python
NEGATIVE_PROMPT = (
    "aerial view, drone, field, landscape, panorama, wide shot, full field, "
    "rows, plot, farm, horizon, watermark, text"
)
```

#### Parámetros Actualizados
| Parámetro | Valor Anterior | Valor Nuevo | Razón |
|-----------|---------------|-------------|-------|
| `guidance_scale` | 7.5 | **7.0** | Optimizado para coherencia plant-centric |
| `LORA_ADAPTER_WEIGHTS` | `[0.3]` | **`[1.0]`** | Máxima influencia de la adaptación LoRA |
| `DEFAULT_SEED` | 1234 | **123** | Alineado con HPC |
| Seeds múltiples | N/A | **[123, 999]** | Para variación controlada |

---

### 2. **Módulo de Inferencia (`src/inference.py`)**

#### Método de Carga LoRA
**Cambio**: De `fuse_lora()` a `load_lora_weights()` + `set_adapters()`

```python
# ANTES (fusión directa)
self.pipe.load_lora_weights(str(self.config.LORA_WEIGHTS_PATH))
self.pipe.fuse_lora(lora_scale=lora_scale)

# AHORA (adaptadores configurables)
self.pipe.load_lora_weights(str(self.config.LORA_WEIGHTS_PATH))
try:
    self.pipe.set_adapters(['default'], adapter_weights=[adapter_weight])
    print(f'[OK] set_adapters(default, {adapter_weight})')
except Exception as e:
    print(f'[WARN] set_adapters not available: {e}')
```

**Ventajas**:
- ✅ Mayor control sobre el peso del adaptador
- ✅ Cambios dinámicos sin recargar modelo
- ✅ Compatible con PEFT backend
- ✅ Manejo robusto de errores con fallback

#### Negative Prompts
Añadido soporte para `negative_prompt` en `generate_image()`:

```python
def generate_image(
    self,
    prompt: str,
    negative_prompt: Optional[str] = None,  # ⬅️ NUEVO
    seed: Optional[int] = None,
    # ... otros parámetros
) -> Image.Image:
```

- Valor por defecto: `InferenceConfig.NEGATIVE_PROMPT`
- Evita características no deseadas (vistas aéreas, campos, etc.)

#### Actualización de Pesos LoRA Simplificada
```python
def update_lora_scale(self, new_scale: float) -> None:
    """Actualiza peso sin recargar pipeline completo."""
    try:
        self.pipe.set_adapters(['default'], adapter_weights=[new_scale])
        self.lora_scale = new_scale
        print(f'[OK] Updated adapter weight to {new_scale}')
    except Exception as e:
        print(f'[WARN] Could not update adapter weight: {e}')
```

---

### 3. **Aplicación Streamlit (`app.py`)**

#### Selector de Prompts Predefinidos
Añadido en Tab 1 (Generación Simple) y Tab 2 (Comparación):

```python
prompt_preset = st.selectbox(
    "Prompt Predefinido",
    options=["Personalizado"] + [f"Preset {i+1}" for i in range(len(InferenceConfig.DEFAULT_PROMPTS))],
    help="Selecciona un prompt plant-centric predefinido o personalizado"
)
```

#### Campo Negative Prompt
```python
negative_prompt_simple = st.text_area(
    "Negative Prompt",
    value=InferenceConfig.NEGATIVE_PROMPT,
    height=60,
    help="Características a evitar en la generación"
)
```

#### Valores por Defecto Actualizados
- **Guidance Scale**: 7.5 → **7.0**
- **Peso LoRA**: 0.3 → **1.0**
- **Rango Peso LoRA**: 0.1-1.0 → **0.0-1.0** (permite desactivar LoRA completamente)

---

### 4. **Nuevo Script Batch (`scripts/generate_comparison_batch.py`)**

Script para generación masiva base vs LoRA, replicando funcionalidad del HPC:

#### Características:
- ✅ Genera imágenes con modelo base (sin LoRA)
- ✅ Genera imágenes con adaptación LoRA
- ✅ Múltiples prompts y seeds
- ✅ Estructura organizada: `outputs/base/` y `outputs/lora/`
- ✅ Barra de progreso con `tqdm`
- ✅ Gestión eficiente de memoria (limpieza entre fases)

#### Uso:
```bash
python scripts/generate_comparison_batch.py
```

#### Estructura de Salida:
```
outputs/
├── base/
│   ├── base_p1_seed123.png
│   ├── base_p1_seed999.png
│   ├── base_p2_seed123.png
│   ├── base_p2_seed999.png
│   └── ...
└── lora/
    ├── lora_p1_seed123.png
    ├── lora_p1_seed999.png
    ├── lora_p2_seed123.png
    ├── lora_p2_seed999.png
    └── ...
```

**Total de imágenes generadas**: 4 prompts × 2 seeds × 2 modelos = **16 imágenes**

---

## 🔧 Compatibilidad con Streamlit

### Correcciones Aplicadas:
- ✅ `use_container_width=True` → `use_column_width=True` (API Streamlit 1.31.0)
- ✅ CORS habilitado en `.streamlit/config.toml` (`enableCORS = true`)

---

## 📊 Comparación: Antiguo vs Nuevo

| Aspecto | Antiguo | Nuevo |
|---------|---------|-------|
| **Enfoque** | Vistas aéreas de campos | Plantas individuales |
| **Prompts** | 1 prompt genérico | 4 prompts especializados |
| **Negative Prompt** | No disponible | Implementado |
| **LoRA Weight** | 0.3 (30%) | 1.0 (100%) |
| **Guidance** | 7.5 | 7.0 |
| **Método LoRA** | `fuse_lora()` | `set_adapters()` |
| **Batch Script** | No disponible | Implementado |
| **Interfaz Web** | Funcional básica | Con presets y negative prompts |

---

## 🚀 Uso Actualizado

### Streamlit Web App
```bash
streamlit run app.py
```

Características nuevas:
- Selector de prompts plant-centric predefinidos
- Campo de negative prompt configurable
- Peso LoRA ajustable de 0.0 a 1.0
- Guidance scale optimizado en 7.0

### Script de Línea de Comandos
```bash
# Generar imagen simple con LoRA
python scripts/generate_images.py

# Generar comparaciones batch (nuevo)
python scripts/generate_comparison_batch.py

# Testear pipeline
python scripts/test_inference.py
```

---

## 🔬 Metodología Actualizada

### Pipeline de Generación:
1. **Carga del modelo**: Stable Diffusion v1.5
2. **Carga de LoRA**: `load_lora_weights()` + `set_adapters()`
3. **Configuración de peso**: `adapter_weights=[1.0]`
4. **Generación**:
   - Prompt: Plant-centric con detalles específicos
   - Negative: Evitar vistas aéreas y elementos no deseados
   - Steps: 30
   - Guidance: 7.0
   - Resolución: 512×512
5. **Post-procesamiento**: Guardado en formato PNG

### Seeds Recomendadas:
- **123**: Seed principal (resultados consistentes)
- **999**: Seed alternativa (mayor variación)

---

## ✅ Testing

Para verificar las actualizaciones:

```bash
# 1. Verificar carga de configuración
python -c "from config import InferenceConfig; print(InferenceConfig.DEFAULT_PROMPTS)"

# 2. Testear pipeline básico
python scripts/test_inference.py

# 3. Generar batch de comparación
python scripts/generate_comparison_batch.py

# 4. Iniciar aplicación web
streamlit run app.py
```

---

## 📝 Notas Técnicas

### Compatibilidad PEFT
El método `set_adapters()` requiere PEFT backend. Si no está disponible, el sistema:
- Captura la excepción
- Imprime warning
- Continúa ejecución con carga básica de LoRA

### Gestión de Memoria
En generación batch:
1. Generación completa de imágenes base
2. Limpieza de pipeline (`del pipe`, `gc.collect()`)
3. Vaciar caché CUDA (`torch.cuda.empty_cache()`)
4. Reinicialización con LoRA
5. Generación de imágenes adaptadas

### Reproducibilidad
Seeds fijas (123, 999) aseguran:
- Resultados consistentes entre ejecuciones
- Comparaciones válidas base vs LoRA
- Facilita debugging y evaluación

---

## 🎓 Prompts Plant-Centric

Los nuevos prompts enfatizan:
- ✅ `single [plant_type] plant` - Planta única
- ✅ `centered subject` - Sujeto centrado
- ✅ `close-up crop` - Enfoque cercano
- ✅ `soil background` - Fondo de suelo
- ✅ `natural light` - Iluminación natural
- ✅ `sharp focus, high detail` - Nitidez y detalle

Y evitan (via negative prompt):
- ❌ `aerial view, drone` - Vistas aéreas
- ❌ `field, landscape, panorama` - Vistas amplias
- ❌ `rows, plot, farm` - Contexto agrícola amplio
- ❌ `watermark, text` - Artefactos indeseados

---

## 📚 Referencias

- Script HPC original: `/lustre/home/inia/crueda/hpc/runs/sample-plantcrop-base-vs-lora.sh`
- Diffusers Documentation: https://huggingface.co/docs/diffusers
- PEFT Documentation: https://huggingface.co/docs/peft

---

**Actualizado por**: Sistema de Inferencia SD + LoRA  
**Versión**: 2.0 (Plant-Centric Update)  
**Compatible con**: Python 3.10+, Streamlit 1.31.0, diffusers 0.26.3, PEFT 0.8.2
