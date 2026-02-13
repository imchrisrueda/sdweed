"""
Aplicación web para generación de imágenes sintéticas agrícolas con Stable Diffusion + LoRA.

Esta aplicación proporciona una interfaz gráfica moderna para la generación de imágenes
sintéticas fotorrealistas de cultivos desde perspectiva aérea, utilizando el sistema
de inferencia basado en Stable Diffusion v1.5 con adaptación LoRA especializada.
"""

import streamlit as st
import sys
from pathlib import Path
import torch
from PIL import Image
import io
import time
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="SD LoRA - Generación de Imágenes Agrícolas",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from src import StableDiffusionInference, validate_cuda_availability
from config import InferenceConfig


# Funciones auxiliares
@st.cache_resource
def initialize_inference_system():
    """
    Inicializa el sistema de inferencia con caché para evitar recargas.
    
    Esta función se ejecuta una sola vez al iniciar la aplicación, manteniendo
    el pipeline en memoria para generaciones subsecuentes.
    """
    return StableDiffusionInference(config=InferenceConfig)


def apply_custom_css():
    """Aplica estilos CSS personalizados para interfaz moderna."""
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            background-color: #4CAF50;
            color: white;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stAlert {
            border-radius: 8px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: 600;
        }
        .info-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def save_generation_history(prompt, seed, lora_weight, guidance_scale, steps):
    """
    Guarda el historial de generaciones en el estado de sesión.
    
    Args:
        prompt: Texto del prompt utilizado.
        seed: Semilla de generación.
        lora_weight: Peso del adaptador LoRA.
        guidance_scale: Factor de guidance.
        steps: Número de pasos de inferencia.
    """
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prompt': prompt,
        'seed': seed,
        'lora_weight': lora_weight,
        'guidance_scale': guidance_scale,
        'steps': steps,
    })


def main():
    """Función principal de la aplicación Streamlit."""
    
    apply_custom_css()
    
    # Header
    st.title("🌾 Generación de Imágenes Agrícolas Sintéticas")
    st.markdown("""
        Sistema de inferencia basado en **Stable Diffusion v1.5** con adaptación **LoRA** 
        para generación de imágenes fotorrealistas de cultivos desde perspectiva aérea.
    """)
    
    # Sidebar - Información del sistema
    with st.sidebar:
        st.header("⚙️ Configuración del Sistema")
        
        # Información de hardware
        device_name, cuda_available = validate_cuda_availability()
        
        if cuda_available:
            st.success(f"✅ GPU: {device_name}")
        else:
            st.warning("⚠️ CPU: Sin aceleración GPU")
        
        st.info(f"""
        **Modelo Base:** SD v1.5  
        **Adaptación:** LoRA especializado  
        **Resolución:** 512×512 px
        """)
        
        st.divider()
        
        # Métricas del sistema
        st.subheader("📊 Estado del Sistema")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dispositivo", "GPU" if cuda_available else "CPU")
        with col2:
            if 'generation_count' in st.session_state:
                st.metric("Generadas", st.session_state.generation_count)
            else:
                st.metric("Generadas", 0)
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎨 Generación Simple", 
        "⚖️ Comparación Base vs LoRA",
        "📚 Galería",
        "ℹ️ Información"
    ])
    
    # Tab 1: Generación Simple
    with tab1:
        st.header("Generación con LoRA")
        st.markdown("Genera imágenes sintéticas utilizando el modelo adaptado con LoRA.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input de prompt
            prompt_simple = st.text_area(
                "Prompt de Generación",
                value=InferenceConfig.DEFAULT_PROMPT,
                height=100,
                help="Descripción textual de la imagen a generar"
            )
            
            # Expander para parámetros avanzados
            with st.expander("🔧 Parámetros Avanzados"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    seed_simple = st.number_input(
                        "Seed",
                        min_value=0,
                        max_value=999999,
                        value=InferenceConfig.DEFAULT_SEED,
                        help="Semilla para reproducibilidad"
                    )
                    
                    guidance_simple = st.slider(
                        "Guidance Scale",
                        min_value=1.0,
                        max_value=20.0,
                        value=7.5,
                        step=0.5,
                        help="Factor de adherencia al prompt"
                    )
                
                with col_b:
                    steps_simple = st.slider(
                        "Pasos de Inferencia",
                        min_value=10,
                        max_value=100,
                        value=30,
                        step=5,
                        help="Número de pasos de denoising"
                    )
                    
                    lora_weight_simple = st.slider(
                        "Peso LoRA",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.3,
                        step=0.05,
                        help="Intensidad de la adaptación LoRA"
                    )
        
        with col2:
            st.markdown("### Vista Previa")
            if 'generated_image_simple' in st.session_state:
                st.image(
                    st.session_state.generated_image_simple,
                    caption="Última generación",
                    use_column_width=True
                )
        
        # Botón de generación
        if st.button("🚀 Generar Imagen", key="gen_simple"):
            with st.spinner("Inicializando pipeline de inferencia..."):
                inference = initialize_inference_system()
                
                # Inicializar pipeline con LoRA si es necesario
                inference.initialize_pipeline(load_lora=True)
                
                # Actualizar peso de LoRA si cambió
                if inference.lora_scale != lora_weight_simple:
                    inference.update_lora_scale(lora_weight_simple)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Generando imagen sintética...")
                start_time = time.time()
                
                # Generar imagen
                image = inference.generate_image(
                    prompt=prompt_simple,
                    seed=int(seed_simple),
                    num_inference_steps=int(steps_simple),
                    guidance_scale=float(guidance_simple),
                )
                
                progress_bar.progress(100)
                elapsed_time = time.time() - start_time
                
                # Guardar en estado de sesión
                st.session_state.generated_image_simple = image
                
                # Actualizar contador
                if 'generation_count' not in st.session_state:
                    st.session_state.generation_count = 0
                st.session_state.generation_count += 1
                
                # Guardar historial
                save_generation_history(
                    prompt_simple, seed_simple, lora_weight_simple,
                    guidance_simple, steps_simple
                )
                
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Imagen generada exitosamente en {elapsed_time:.2f}s")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error durante la generación: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        
        # Botones de descarga
        if 'generated_image_simple' in st.session_state:
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # Convertir imagen a bytes para descarga
                buf = io.BytesIO()
                st.session_state.generated_image_simple.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 Descargar Imagen",
                    data=byte_im,
                    file_name=f"synthetic_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                )
            
            with col_d2:
                if st.button("🗑️ Limpiar Resultado"):
                    del st.session_state.generated_image_simple
                    st.rerun()
    
    # Tab 2: Comparación
    with tab2:
        st.header("Comparación Base vs LoRA")
        st.markdown("Genera imágenes comparativas entre el modelo base y el adaptado con LoRA.")
        
        prompt_compare = st.text_area(
            "Prompt de Generación",
            value=InferenceConfig.DEFAULT_PROMPT,
            height=100,
            key="prompt_compare"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            seed_compare = st.number_input(
                "Seed",
                min_value=0,
                max_value=999999,
                value=InferenceConfig.DEFAULT_SEED,
                key="seed_compare"
            )
        
        with col2:
            lora_weight_compare = st.slider(
                "Peso LoRA",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="lora_weight_compare"
            )
        
        if st.button("🔄 Generar Comparación", key="gen_compare"):
            with st.spinner("Generando imágenes comparativas..."):
                inference = initialize_inference_system()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Generar imagen base
                    status_text.text("Generando imagen base (sin LoRA)...")
                    progress_bar.progress(25)
                    
                    inference.initialize_pipeline(load_lora=False)
                    img_base = inference.generate_image(
                        prompt=prompt_compare,
                        seed=int(seed_compare),
                    )
                    
                    progress_bar.progress(50)
                    
                    # Generar imagen con LoRA
                    status_text.text("Generando imagen con adaptación LoRA...")
                    
                    inference.initialize_pipeline(load_lora=True)
                    
                    # Actualizar peso de LoRA si es necesario
                    if inference.lora_scale != lora_weight_compare:
                        inference.update_lora_scale(lora_weight_compare)
                    
                    img_lora = inference.generate_image(
                        prompt=prompt_compare,
                        seed=int(seed_compare),
                    )
                    
                    progress_bar.progress(100)
                    
                    # Guardar en estado
                    st.session_state.comparison_base = img_base
                    st.session_state.comparison_lora = img_lora
                    
                    # Actualizar contador
                    if 'generation_count' not in st.session_state:
                        st.session_state.generation_count = 0
                    st.session_state.generation_count += 2
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success("✅ Comparación generada exitosamente")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error durante la generación: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
        
        # Mostrar comparación
        if 'comparison_base' in st.session_state and 'comparison_lora' in st.session_state:
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Modelo Base (SD v1.5)")
                st.image(
                    st.session_state.comparison_base,
                    caption="Sin adaptación LoRA",
                    use_column_width=True
                )
                
                # Botón de descarga
                buf_base = io.BytesIO()
                st.session_state.comparison_base.save(buf_base, format='PNG')
                st.download_button(
                    label="💾 Descargar Base",
                    data=buf_base.getvalue(),
                    file_name=f"base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    key="download_base"
                )
            
            with col2:
                st.subheader("Modelo con LoRA")
                st.image(
                    st.session_state.comparison_lora,
                    caption=f"Con LoRA (peso: {lora_weight_compare})",
                    use_column_width=True
                )
                
                # Botón de descarga
                buf_lora = io.BytesIO()
                st.session_state.comparison_lora.save(buf_lora, format='PNG')
                st.download_button(
                    label="💾 Descargar LoRA",
                    data=buf_lora.getvalue(),
                    file_name=f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    key="download_lora"
                )
    
    # Tab 3: Galería
    with tab3:
        st.header("📚 Historial de Generaciones")
        
        if 'history' in st.session_state and len(st.session_state.history) > 0:
            st.markdown(f"**Total de generaciones:** {len(st.session_state.history)}")
            
            # Mostrar historial en tabla
            for idx, entry in enumerate(reversed(st.session_state.history[-10:])):
                with st.expander(f"Generación {len(st.session_state.history) - idx} - {entry['timestamp']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.text_area("Prompt", entry['prompt'], height=80, disabled=True, key=f"prompt_{idx}")
                    
                    with col2:
                        st.metric("Seed", entry['seed'])
                        st.metric("LoRA Weight", f"{entry['lora_weight']:.2f}")
                        st.metric("Guidance", f"{entry['guidance_scale']:.1f}")
                        st.metric("Steps", entry['steps'])
        else:
            st.info("No hay generaciones en el historial. Comienza generando imágenes en las pestañas anteriores.")
    
    # Tab 4: Información
    with tab4:
        st.header("ℹ️ Información del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Arquitectura Técnica")
            st.markdown("""
            **Modelo Base:**
            - Stable Diffusion v1.5
            - Modelo de difusión probabilística latente
            - Resolución: 512×512 píxeles
            
            **Adaptación LoRA:**
            - Low-Rank Adaptation entrenada en HPC
            - Especialización en imágenes agrícolas aéreas
            - Control morfológico mejorado
            
            **Pipeline de Inferencia:**
            1. Codificación del prompt (CLIP)
            2. Inicialización de latentes (ruido gaussiano)
            3. Denoising iterativo (UNet + LoRA)
            4. Decodificación VAE
            """)
        
        with col2:
            st.subheader("Parámetros Principales")
            st.markdown("""
            **Guidance Scale:**
            - Controla adherencia al prompt
            - Rango: 1.0 - 20.0
            - Recomendado: 7.0 - 9.0
            
            **Pasos de Inferencia:**
            - Iteraciones de denoising
            - Más pasos = mayor calidad
            - Balance: calidad vs velocidad
            
            **Peso LoRA:**
            - Intensidad de adaptación
            - Rango: 0.1 - 1.0
            - Recomendado: 0.3 - 0.5
            
            **Seed:**
            - Garantiza reproducibilidad
            - Mismo seed = misma imagen
            """)
        
        st.divider()
        
        st.subheader("📖 Documentación Adicional")
        st.markdown("""
        - **README.md**: Documentación completa del proyecto
        - **docs/usage.md**: Guía detallada de uso
        - **docs/methodology.md**: Fundamentos técnicos
        """)
        
        st.divider()
        
        st.subheader("🔬 Problema Abordado")
        st.markdown("""
        Generación de imágenes sintéticas fotorrealistas, estructuralmente coherentes y 
        semánticamente controlables de cultivos objetivo (sorghum), vistas desde un plano 
        aproximadamente paralelo al suelo. Las imágenes preservan características morfológicas 
        esenciales observadas en imágenes reales de campo y amplían sistemáticamente el espacio 
        de variabilidad visual, con el fin de mejorar la generalización de modelos de visión 
        por computador entrenados con datos aumentados.
        """)


if __name__ == "__main__":
    main()
