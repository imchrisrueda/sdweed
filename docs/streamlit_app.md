# Guía de la Aplicación Web - Generación de Imágenes Sintéticas

## Descripción General

La aplicación web desarrollada en Streamlit proporciona una interfaz gráfica moderna e intuitiva para la generación de imágenes sintéticas agrícolas utilizando Stable Diffusion v1.5 con adaptación LoRA. Esta interfaz elimina la necesidad de interacción por línea de comandos, democratizando el acceso al sistema de inferencia.

## Iniciar la Aplicación

### Método 1: Inicio Directo
```bash
streamlit run app.py
```

### Método 2: Con Configuración Personalizada
```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

La aplicación se abrirá automáticamente en el navegador predeterminado. Si no se abre automáticamente, acceder manualmente a:
- **Local**: http://localhost:8501
- **Red**: http://<ip-del-servidor>:8501

## Arquitectura de la Aplicación

### Componentes Principales

```
app.py
├── Main Layout (Diseño principal)
│   ├── Header con información del sistema
│   ├── Sidebar con estado y métricas
│   └── Tabs para diferentes funcionalidades
│
├── Tab 1: Generación Simple
│   ├── Input de prompt
│   ├── Controles de parámetros
│   ├── Vista previa de imagen
│   └── Opciones de descarga
│
├── Tab 2: Comparación Base vs LoRA
│   ├── Generación paralela
│   ├── Visualización lado a lado
│   └── Análisis comparativo
│
├── Tab 3: Galería e Historial
│   ├── Registro de generaciones
│   ├── Parámetros utilizados
│   └── Métricas de sesión
│
└── Tab 4: Información del Sistema
    ├── Arquitectura técnica
    ├── Documentación de parámetros
    └── Referencias
```

## Funcionalidades Detalladas

### 1. Generación Simple 🎨

Esta pestaña permite la generación rápida de imágenes individuales con control completo de parámetros.

#### Controles Básicos

**Prompt de Generación**
- Campo de texto expandible para descripción detallada
- Valor predeterminado optimizado para imágenes agrícolas
- Sugerencias automáticas basadas en prompts efectivos

**Generación Inmediata**
- Botón de generación prominente
- Feedback visual durante el proceso (barra de progreso)
- Estimación de tiempo basada en hardware detectado

#### Parámetros Avanzados (Expandible)

Accesibles mediante el expander "🔧 Parámetros Avanzados":

**Seed (Semilla)**
- Rango: 0 - 999,999
- Garantiza reproducibilidad exacta
- Mismo seed con mismos parámetros = imagen idéntica
- Útil para experimentos controlados

**Guidance Scale**
- Rango: 1.0 - 20.0
- Controla adherencia al prompt
- Valores bajos (1-5): Mayor creatividad, menor fidelidad
- Valores medios (7-9): Balance óptimo (recomendado)
- Valores altos (10-20): Máxima fidelidad, riesgo de artefactos

**Pasos de Inferencia**
- Rango: 10 - 100
- Número de iteraciones de denoising
- Más pasos = mayor calidad pero mayor tiempo
- Recomendado: 30-50 pasos

**Peso LoRA**
- Rango: 0.1 - 1.0
- Intensidad de la adaptación especializada
- 0.1-0.2: Influencia sutil
- 0.3-0.5: Balance óptimo (recomendado)
- 0.6-1.0: Fuerte especialización

#### Vista Previa y Descarga

- Visualización inmediata de la imagen generada
- Botón de descarga con nombre automático con timestamp
- Formato PNG con calidad máxima
- Opción de limpiar resultado para nueva generación

#### Ejemplo de Flujo de Trabajo

1. Ingresar o modificar el prompt descriptivo
2. Ajustar parámetros según necesidad (opcional)
3. Clic en "🚀 Generar Imagen"
4. Observar progreso en tiempo real
5. Visualizar resultado en panel lateral
6. Descargar imagen si satisface requisitos
7. Ajustar parámetros y regenerar si es necesario

### 2. Comparación Base vs LoRA ⚖️

Esta funcionalidad permite evaluar visualmente el impacto de la adaptación LoRA mediante generación paralela.

#### Proceso de Comparación

1. **Configuración del Experimento**
   - Prompt único aplicado a ambos modelos
   - Seed compartido para comparación justa
   - Peso de LoRA configurable

2. **Generación Secuencial**
   - Fase 1: Generación con modelo base (SD v1.5 puro)
   - Fase 2: Generación con modelo adaptado (SD v1.5 + LoRA)
   - Progreso visual para cada fase

3. **Visualización Comparativa**
   - Presentación lado a lado
   - Misma escala y dimensiones
   - Etiquetas descriptivas para cada versión

#### Análisis Cualitativo

Al comparar resultados, evaluar:

**Coherencia Estructural**
- Regularidad en patrones de distribución espacial
- Consistencia en escala de elementos vegetales
- Organización geométrica de plantas

**Fidelidad Morfológica**
- Características foliares específicas del cultivo
- Textura de suelo y superficie
- Proporciones realistas

**Control Semántico**
- Respuesta a términos técnicos del prompt
- Diferenciación de estadíos fenológicos
- Interpretación de atributos ambientales

#### Descarga Independiente

- Botones de descarga separados para cada versión
- Nomenclatura clara: `base_*.png` y `lora_*.png`
- Timestamps para trazabilidad temporal

### 3. Galería e Historial 📚

Sistema de registro automático de todas las generaciones realizadas durante la sesión.

#### Información Registrada

Para cada generación se almacena:
- **Timestamp**: Fecha y hora exacta
- **Prompt**: Texto completo utilizado
- **Seed**: Valor de la semilla
- **LoRA Weight**: Peso del adaptador
- **Guidance Scale**: Factor de guidance
- **Steps**: Número de pasos de inferencia

#### Visualización del Historial

- Vista en orden cronológico inverso (más reciente primero)
- Últimas 10 generaciones visibles
- Expanders individuales para cada entrada
- Formato organizado con métricas destacadas

#### Utilidad

- **Reproducibilidad**: Acceso rápido a parámetros exitosos
- **Experimentación**: Comparación de configuraciones previas
- **Documentación**: Registro automático de experimentos
- **Análisis**: Evaluación de patrones en parámetros utilizados

### 4. Información del Sistema ℹ️

Documentación integrada y estado del hardware en tiempo real.

#### Arquitectura Técnica

Resumen de:
- Especificaciones del modelo base
- Detalles de la adaptación LoRA
- Pipeline de inferencia paso a paso
- Fundamentos del proceso de generación

#### Documentación de Parámetros

Explicación detallada de:
- Guidance Scale y su impacto
- Número de pasos de inferencia
- Peso del adaptador LoRA
- Seed y reproducibilidad

#### Enlaces a Documentación

Acceso directo a:
- README.md principal
- Guía de uso detallada (docs/usage.md)
- Metodología técnica (docs/methodology.md)

## Sidebar: Estado del Sistema

### Información de Hardware

**Con GPU NVIDIA Disponible:**
- Indicador verde de estado ✅
- Nombre del modelo de GPU
- Tipo de dato: float16 (optimizado)

**Solo CPU Disponible:**
- Advertencia amarilla ⚠️
- Notificación de inferencia lenta
- Tipo de dato: float32

### Métricas en Tiempo Real

**Contador de Generaciones**
- Actualización automática con cada generación
- Persiste durante toda la sesión
- Útil para seguimiento de productividad

## Optimizaciones Implementadas

### Cache de Recursos

La decoración `@st.cache_resource` en `initialize_inference_system()` garantiza:
- Carga única del modelo al iniciar la aplicación
- Reutilización del pipeline para todas las generaciones
- Eliminación de tiempo de carga repetitivo
- Uso eficiente de memoria GPU/RAM

### Gestión de Estado

Uso de `st.session_state` para:
- Persistencia de imágenes generadas entre interacciones
- Mantenimiento del historial durante la sesión
- Contador de generaciones acumulativas
- Configuraciones temporales de usuario

### Feedback Visual

- Barras de progreso durante inferencia
- Spinners para operaciones de carga
- Mensajes de éxito/error contextuales
- Tiempo de generación reportado

## Personalización de la Interfaz

### CSS Personalizado

La función `apply_custom_css()` implementa:
- Esquema de colores profesional
- Botones con colores consistentes
- Bordes redondeados y sombras suaves
- Espaciado optimizado para legibilidad

### Configuración de Página

Parámetros de `st.set_page_config()`:
- **Título**: Mostrado en pestaña del navegador
- **Ícono**: Emoji representativo (🌾)
- **Layout**: Wide para aprovechar espacio horizontal
- **Sidebar**: Expandido por defecto

## Casos de Uso Recomendados

### Caso 1: Exploración Inicial

**Objetivo**: Familiarizarse con capacidades del sistema

1. Usar Tab "Generación Simple"
2. Mantener parámetros predeterminados
3. Generar con prompt por defecto
4. Experimentar con variaciones de seed
5. Observar diversidad en resultados

### Caso 2: Ajuste de Prompts

**Objetivo**: Optimizar descripción textual para resultados deseados

1. Comenzar con prompt base
2. Añadir términos técnicos específicos
3. Especificar estadíos fenológicos
4. Incluir condiciones ambientales
5. Evaluar respuesta del modelo

### Caso 3: Evaluación de LoRA

**Objetivo**: Cuantificar impacto de adaptación

1. Usar Tab "Comparación Base vs LoRA"
2. Prompt consistente para ambas generaciones
3. Seed fijo para comparación justa
4. Variar peso de LoRA (0.1, 0.3, 0.5)
5. Comparar coherencia y fidelidad

### Caso 4: Generación Batch

**Objetivo**: Producir conjunto de imágenes para dataset

1. Definir prompt estándar optimizado
2. Fijar parámetros de calidad (steps: 40, guidance: 7.5)
3. Variar seed sistemáticamente
4. Descargar todas las generaciones
5. Organizar por características observadas

## Solución de Problemas

### Aplicación No Responde

**Síntoma**: Interfaz congelada durante generación

**Causa**: Normal durante inferencia intensiva

**Solución**: Esperar finalización del proceso (monitor barra de progreso)

### Error: CUDA Out of Memory

**Síntoma**: Mensaje de error sobre memoria GPU

**Soluciones**:
1. Cerrar otras aplicaciones que usen GPU
2. Reducir pasos de inferencia a 20-25
3. Reiniciar aplicación para limpiar caché
4. Ejecutar en CPU (más lento pero funcional)

### Imágenes No se Generan

**Síntoma**: Error durante proceso de generación

**Verificaciones**:
1. Pesos LoRA presentes en `models/lora_weights/`
2. Dependencias instaladas correctamente
3. Prompt no excede límite de tokens (77)
4. Conexión a internet para descarga de modelo base

### Aplicación Lenta en CPU

**Síntoma**: Generación toma >5 minutos

**Esperado**: Inferencia en CPU es significativamente más lenta

**Recomendaciones**:
1. Reducir pasos de inferencia a 15-20
2. Ejecutar en horario de baja actividad
3. Considerar uso de GPU mediante servicio cloud

## Extensibilidad

### Añadir Nuevos Prompts Predefinidos

Modificar en `app.py`:

```python
PRESET_PROMPTS = {
    "Sorghum Temprano": "aerial drone photograph, top-down view...",
    "Sorghum Maduro": "aerial view, mature sorghum plants...",
    "Campo Denso": "dense agricultural field, sorghum...",
}

selected_preset = st.selectbox("Prompts Predefinidos", PRESET_PROMPTS.keys())
prompt = PRESET_PROMPTS[selected_preset]
```

### Integrar Métricas de Calidad

Añadir evaluación automática de imágenes generadas:

```python
from src.metrics import calculate_fid, calculate_clip_score

if image_generated:
    fid_score = calculate_fid(image, reference_images)
    clip_score = calculate_clip_score(image, prompt)
    
    st.metric("FID Score", f"{fid_score:.2f}")
    st.metric("CLIP Score", f"{clip_score:.3f}")
```

### Exportar Historial a CSV

Implementar función de exportación:

```python
import pandas as pd

if st.button("Exportar Historial"):
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False)
    st.download_button(
        "Descargar CSV",
        csv,
        "historial_generaciones.csv",
        "text/csv"
    )
```

## Consideraciones de Despliegue

### Despliegue Local

Ideal para:
- Uso individual en estación de trabajo
- Experimentos en clúster HPC con GPU dedicada
- Desarrollo y pruebas

### Despliegue en Red Local

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Permite acceso desde otros dispositivos en la misma red.

### Despliegue Cloud

Plataformas recomendadas:
- **Streamlit Cloud**: Hosting gratuito para apps públicas
- **Hugging Face Spaces**: GPU gratuita temporal
- **AWS/GCP/Azure**: Instancias con GPU para producción

**Nota**: Despliegue cloud requiere consideraciones adicionales de seguridad y autenticación.

## Mejores Prácticas

### Optimización de Prompts

1. Ser específico y descriptivo
2. Incluir términos técnicos relevantes
3. Especificar perspectiva de captura
4. Mencionar condiciones ambientales
5. Indicar estadío fenológico del cultivo

### Uso Eficiente de Recursos

1. Cerrar aplicación al finalizar sesión
2. Utilizar cache de modelo (no reiniciar innecesariamente)
3. Ajustar pasos de inferencia según necesidad
4. Descargar imágenes importantes antes de cerrar

### Experimentación Sistemática

1. Documentar parámetros en historial
2. Variar un parámetro a la vez
3. Mantener seed fijo para comparaciones
4. Evaluar resultados cualitativamente
5. Organizar descargas en carpetas descriptivas

## Referencias Técnicas

- **Streamlit Documentation**: https://docs.streamlit.io
- **Diffusers Library**: https://huggingface.co/docs/diffusers
- **Stable Diffusion**: https://stability.ai/stable-diffusion
- **LoRA Paper**: https://arxiv.org/abs/2106.09685

---

Para consultas técnicas avanzadas o problemas específicos, consultar la documentación principal en [`README.md`](../README.md) o la guía metodológica en [`methodology.md`](methodology.md).
