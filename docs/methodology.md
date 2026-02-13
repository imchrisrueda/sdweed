# Metodología Técnica - Generación de Imágenes Sintéticas Agrícolas

## Fundamentos Teóricos

### Modelos de Difusión Probabilística

Los modelos de difusión constituyen una clase de modelos generativos que aprenden a revertir un proceso gradual de adición de ruido gaussiano. El modelo Stable Diffusion implementa este paradigma en el espacio latente comprimido de un autoencoder variacional (VAE).

#### Proceso Forward (Difusión)

Dado un punto de datos $x_0 \sim q(x_0)$, el proceso de difusión añade ruido gaussiano en $T$ pasos:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

donde $\beta_t$ es un schedule de varianza que controla la magnitud del ruido añadido en cada paso.

#### Proceso Reverse (Generación)

El modelo aprende a revertir este proceso, estimando $p_\theta(x_{t-1}|x_t)$:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Durante la inferencia, se parte de ruido puro $x_T \sim \mathcal{N}(0, I)$ y se aplican $T$ pasos de denoising para obtener $x_0$.

### Latent Diffusion Models (LDM)

Stable Diffusion opera en el espacio latente de un VAE pre-entrenado, reduciendo la dimensionalidad computacional:

$$z = \mathcal{E}(x), \quad x' = \mathcal{D}(z)$$

donde $\mathcal{E}$ y $\mathcal{D}$ son el encoder y decoder del VAE respectivamente. El proceso de difusión se aplica sobre $z$ en lugar de $x$, reduciendo el costo computacional en un factor de $\sim$64.

### Text-Conditional Generation

El condicionamiento textual se implementa mediante cross-attention entre las representaciones latentes de la imagen y los embeddings del prompt obtenidos de un text encoder (CLIP):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

donde $Q$ proviene de las features visuales y $K, V$ de las representaciones textuales.

### Classifier-Free Guidance

El parámetro `guidance_scale` implementa classifier-free guidance, amplificando la influencia del condicionamiento textual:

$$\tilde{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \emptyset) + s \cdot (\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset))$$

donde $s$ es el guidance scale, $c$ es el condicionamiento textual y $\emptyset$ representa generación no condicionada.

## Adaptación LoRA

### Fundamento de Low-Rank Adaptation

LoRA introduce matrices de bajo rango en las capas de atención del UNet, permitiendo fine-tuning eficiente sin modificar los pesos originales del modelo:

$$W' = W_0 + \Delta W = W_0 + BA$$

donde:
- $W_0 \in \mathbb{R}^{d \times k}$: Pesos originales pre-entrenados (congelados)
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: Matrices de adaptación de rango $r$
- $r \ll \min(d, k)$: Rango de la descomposición (típicamente 4-64)

Esta factorización reduce drásticamente el número de parámetros entrenables:

$$\text{Params}(\Delta W) = r(d + k) \ll dk = \text{Params}(W_0)$$

### Integración en el Pipeline

Durante la inferencia, los pesos LoRA se combinan con los originales mediante un factor de escala $\alpha$:

$$W_{\text{effective}} = W_0 + \alpha \cdot BA$$

El parámetro `adapter_weights` en la configuración corresponde a $\alpha$, controlando la magnitud de la adaptación.

### Ventajas para Agricultura de Precisión

1. **Eficiencia paramétrica**: Entrenamiento de ~1M parámetros vs ~860M del modelo completo
2. **Preservación de conocimiento general**: Los pesos base mantienen capacidades generalistas
3. **Especialización de dominio**: Los pesos LoRA capturan características morfológicas específicas de cultivos
4. **Transferibilidad**: Los pesos LoRA pueden aplicarse sobre diferentes checkpoints base

## Pipeline de Inferencia

### Etapa 1: Codificación del Prompt

```
Prompt textual → CLIP Text Encoder → Embeddings textuales (77 tokens × 768 dims)
```

El text encoder transforma el prompt en representaciones vectoriales que guiarán el proceso de generación.

### Etapa 2: Inicialización de Latentes

```
Ruido gaussiano N(0,I) → Latentes iniciales (4 × 64 × 64)
```

Se genera un tensor latente de ruido puro con dimensiones reducidas respecto a la imagen final.

### Etapa 3: Denoising Iterativo

Para cada timestep $t$ desde $T$ hasta 0:

1. **UNet Prediction**: El UNet con LoRA predice el ruido residual condicionado al timestep y al prompt
2. **Scheduler Step**: Se aplica el paso del scheduler (PNDM, DDIM, Euler, etc.) para actualizar los latentes
3. **Guidance Application**: Se aplica classifier-free guidance si `guidance_scale > 1`

### Etapa 4: Decodificación VAE

```
Latentes finales (4 × 64 × 64) → VAE Decoder → Imagen RGB (3 × 512 × 512)
```

El decoder VAE transforma los latentes refinados al espacio de píxeles, generando la imagen final.

## Parámetros de Generación

### num_inference_steps

Controla el número de pasos de denoising. Mayor número de pasos:
- **Ventaja**: Convergencia más refinada, mayor calidad visual
- **Desventaja**: Mayor tiempo de cómputo

Relación empírica: $\text{Tiempo} \propto \text{num\_steps}$

### guidance_scale

Factor de amplificación del condicionamiento textual. Valores típicos:
- **1.0**: No guidance (generación no condicionada)
- **7.0-9.0**: Balance óptimo para generación realista
- **15.0+**: Máxima adherencia al prompt, riesgo de sobresaturación

### Seed

Inicializa el generador de números pseudoaleatorios, garantizando reproducibilidad determinística bajo condiciones idénticas de hardware y software.

## Consideraciones de Calidad

### Coherencia Estructural

La adaptación LoRA mejora la coherencia morfológica mediante:
1. **Regularización geométrica**: Aprendizaje de relaciones espaciales características de vistas aéreas
2. **Consistencia textural**: Captura de patrones de textura de suelo y vegetación
3. **Control de escala**: Preservación de proporciones realistas entre elementos de la escena

### Fidelidad Semántica

La efectividad del condicionamiento textual depende de:
- **Especificidad del prompt**: Términos técnicos vs descripciones genéricas
- **Alineación con distribución de entrenamiento**: Conceptos vistos durante pre-training
- **Composicionalidad**: Capacidad del modelo de combinar múltiples atributos

### Variabilidad Sistemática

La generación con múltiples seeds permite:
- **Exploración del espacio latente**: Muestreo de diferentes modos de la distribución aprendida
- **Quantificación de incertidumbre**: Evaluación de consistencia entre generaciones
- **Aumentación de datos**: Expansión del conjunto de entrenamiento con variaciones controladas

## Limitaciones Técnicas

### Resolución Limitada

El modelo base está optimizado para 512×512 píxeles. Generación a resoluciones superiores puede producir:
- Repetición de patrones estructurales
- Inconsistencias en elementos de detalle fino
- Artefactos en bordes de regiones semánticas

### Sesgo de Distribución

El modelo hereda sesgos de su conjunto de entrenamiento (LAION-5B):
- Predominancia de perspectivas terrestres vs aéreas
- Distribución geográfica no uniforme de imágenes agrícolas
- Mayor representación de cultivos de zonas templadas

### Coherencia Temporal

Las generaciones son independientes entre sí. No existe garantía de coherencia al interpolar seeds o modificar prompts gradualmente.

## Métricas de Evaluación

### Métricas Cuantitativas

1. **Frechet Inception Distance (FID)**: Mide la similitud entre distribuciones de imágenes reales y sintéticas
2. **Inception Score (IS)**: Evalúa diversidad y calidad mediante clasificador pre-entrenado
3. **CLIP Score**: Mide alineación semántica entre imagen generada y prompt

### Evaluación Cualitativa

1. **Fotorrealismo**: Inspección visual de coherencia textural y luminosidad
2. **Fidelidad morfológica**: Comparación con características de imágenes reales de campo
3. **Diversidad**: Análisis de variabilidad en múltiples generaciones con mismo prompt

## Referencias Bibliográficas

1. Ho, J. et al. (2020). "Denoising Diffusion Probabilistic Models". NeurIPS.
2. Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models". CVPR.
3. Hu, E. J. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models". ICLR.
4. Schuhmann, C. et al. (2022). "LAION-5B: An open large-scale dataset for training next generation image-text models". NeurIPS.
5. Nichol, A. et al. (2021). "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models". ICML.

## Aplicaciones en Agricultura de Precisión

### Aumentación de Datos para Computer Vision

Las imágenes sintéticas pueden utilizarse para:
- **Entrenamiento de detectores de cultivos**: Complemento a datasets reales limitados
- **Simulación de condiciones adversas**: Generación de escenarios subrepresentados
- **Balanceo de clases**: Síntesis de instancias de clases minoritarias

### Planificación Experimental

Generación de escenarios hipotéticos para:
- Evaluación de algoritmos de segmentación bajo condiciones variadas
- Diseño de patrones de siembra y su detectabilidad
- Simulación de estadíos fenológicos intermedios

### Validación de Modelos

Uso de imágenes sintéticas con ground truth conocido para:
- Análisis de sensibilidad de modelos de detección
- Cuantificación de degradación de performance bajo variabilidad controlada
- Identificación de failure modes en condiciones extremas
