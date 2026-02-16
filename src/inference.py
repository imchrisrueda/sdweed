"""
Módulo de inferencia para la generación de imágenes sintéticas con Stable Diffusion + LoRA.

Este módulo encapsula la lógica del pipeline de generación, permitiendo la creación
de imágenes tanto con el modelo base como con adaptaciones LoRA específicas para
la síntesis de imágenes agrícolas fotorrealistas.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from config import InferenceConfig


class StableDiffusionInference:
    """
    Clase principal para inferencia con Stable Diffusion y adaptadores LoRA.
    
    Esta clase gestiona la inicialización del pipeline de difusión, la carga de 
    pesos LoRA y la generación de imágenes sintéticas con parámetros configurables.
    
    Attributes:
        config: Instancia de InferenceConfig con parámetros del sistema.
        pipe: Pipeline de Diffusers para Stable Diffusion.
        lora_loaded: Indicador de carga exitosa de pesos LoRA.
    """
    
    def __init__(self, config: InferenceConfig = InferenceConfig):
        """
        Inicializa el sistema de inferencia.
        
        Args:
            config: Objeto de configuración con parámetros del sistema.
        """
        self.config = config
        self.config.validate_paths()
        self.pipe = None
        self.lora_loaded = False
        self.lora_scale = None
        
    def initialize_pipeline(self, load_lora: bool = False) -> None:
        """
        Inicializa el pipeline de Stable Diffusion.
        
        Carga el modelo base desde HuggingFace Hub y opcionalmente los pesos LoRA.
        Se deshabilita el safety_checker para permitir generación sin restricciones
        de contenido, dado que se trabaja con imágenes científicas agrícolas.
        
        Args:
            load_lora: Si True, carga los pesos LoRA después del modelo base.
        
        Nota: Si el pipeline ya está inicializado y load_lora cambia, se reiniciará.
        """
        # Si el pipeline ya existe y el estado de LoRA es el mismo, no reinicializar
        if self.pipe is not None and self.lora_loaded == load_lora:
            return
        
        # Si necesitamos cambiar el estado de LoRA, limpiar el pipeline existente
        if self.pipe is not None:
            del self.pipe
            import gc
            gc.collect()
            if self.config.DEVICE == "cuda":
                import torch
                torch.cuda.empty_cache()
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config.MODEL_ID,
            torch_dtype=self.config.DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.to(self.config.DEVICE)
        self.pipe.set_progress_bar_config(disable=True)
        
        if load_lora:
            self._load_lora_weights()
    
    def _load_lora_weights(self) -> None:
        """
        Carga los pesos LoRA entrenados en el pipeline.
        
        Los pesos LoRA han sido entrenados específicamente para adaptar el modelo
        base a la generación de imágenes agrícolas con características morfológicas
        coherentes y control semántico mejorado.
        
        Este método utiliza load_lora_weights() seguido de set_adapters() para
        controlar el peso de influencia del adaptador LoRA.
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline debe ser inicializado antes de cargar LoRA")
        
        # Cargar pesos LoRA desde archivo
        self.pipe.load_lora_weights(str(self.config.LORA_WEIGHTS_PATH))
        
        # Configurar peso del adaptador LoRA (1.0 = máxima influencia)
        adapter_weight = self.config.LORA_ADAPTER_WEIGHTS[0]
        try:
            self.pipe.set_adapters(['default'], adapter_weights=[adapter_weight])
            print(f'[OK] set_adapters(default, {adapter_weight})')
        except Exception as e:
            print(f'[WARN] set_adapters not available: {e}')
        
        self.lora_loaded = True
        self.lora_scale = adapter_weight
    
    def update_lora_scale(self, new_scale: float) -> None:
        """
        Actualiza el peso de influencia de LoRA sin recargar todo el pipeline.
        
        Args:
            new_scale: Nuevo peso de LoRA (típicamente entre 0.0 y 1.0).
        
        Raises:
            RuntimeError: Si el pipeline no ha sido inicializado con LoRA.
        """
        if not self.lora_loaded:
            raise RuntimeError("LoRA no está cargado. Inicialice el pipeline con load_lora=True primero.")
        
        if self.lora_scale == new_scale:
            return  # No es necesario actualizar
        
        # Actualizar peso del adaptador
        try:
            self.pipe.set_adapters(['default'], adapter_weights=[new_scale])
            self.lora_scale = new_scale
            self.config.LORA_ADAPTER_WEIGHTS = [new_scale]
            print(f'[OK] Updated adapter weight to {new_scale}')
        except Exception as e:
            print(f'[WARN] Could not update adapter weight: {e}')
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Image.Image:
        """
        Genera una imagen sintética basada en el prompt proporcionado.
        
        El proceso de generación utiliza el método de difusión probabilística,
        refinando iterativamente el ruido aleatorio hacia una imagen coherente
        que satisface las restricciones semánticas del prompt.
        
        Args:
            prompt: Descripción textual de la imagen a generar.
            negative_prompt: Características a evitar en la generación (opcional).
            seed: Semilla para reproducibilidad (opcional).
            num_inference_steps: Número de pasos de denoising (opcional).
            guidance_scale: Factor de guiado classifier-free (opcional).
            height: Altura de la imagen en píxeles (opcional).
            width: Ancho de la imagen en píxeles (opcional).
        
        Returns:
            Imagen PIL generada.
        
        Raises:
            RuntimeError: Si el pipeline no ha sido inicializado.
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline no inicializado. Ejecute initialize_pipeline() primero.")
        
        # Usar valores por defecto si no se especifican
        negative_prompt = negative_prompt or self.config.NEGATIVE_PROMPT
        num_inference_steps = num_inference_steps or self.config.DEFAULT_INFERENCE_PARAMS["num_inference_steps"]
        guidance_scale = guidance_scale or self.config.DEFAULT_INFERENCE_PARAMS["guidance_scale"]
        height = height or self.config.DEFAULT_INFERENCE_PARAMS["height"]
        width = width or self.config.DEFAULT_INFERENCE_PARAMS["width"]
        seed = seed if seed is not None else self.config.DEFAULT_SEED
        
        # Configurar generador para reproducibilidad
        generator = torch.Generator(device=self.config.DEVICE).manual_seed(seed)
        
        # Ejecutar pipeline de inferencia
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )
        
        return output.images[0]
    
    def generate_comparison(
        self,
        prompt: str,
        seed: int,
        output_prefix: str = "comparison",
    ) -> Tuple[Path, Path]:
        """
        Genera imágenes comparativas con y sin adaptación LoRA.
        
        Este método permite evaluar visualmente el impacto del fine-tuning LoRA
        sobre la generación base, facilitando el análisis cualitativo de la
        mejora en coherencia estructural y fidelidad morfológica.
        
        Args:
            prompt: Descripción textual para generación.
            seed: Semilla para reproducibilidad.
            output_prefix: Prefijo para nombres de archivos de salida.
        
        Returns:
            Tupla con rutas a las imágenes generadas (base, lora).
        """
        # Generar imagen base (sin LoRA)
        self.initialize_pipeline(load_lora=False)
        img_base = self.generate_image(prompt, seed)
        path_base = self.config.OUTPUT_DIR / f"{output_prefix}_base.png"
        img_base.save(path_base)
        
        # Generar imagen con LoRA
        self.initialize_pipeline(load_lora=True)
        img_lora = self.generate_image(prompt, seed)
        path_lora = self.config.OUTPUT_DIR / f"{output_prefix}_lora_w{self.config.LORA_ADAPTER_WEIGHTS[0]}.png"
        img_lora.save(path_lora)
        
        return path_base, path_lora
