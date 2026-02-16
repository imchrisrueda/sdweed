"""
Configuración centralizada para la inferencia de imágenes sintéticas con Stable Diffusion + LoRA.

Este módulo centraliza los parámetros de configuración del pipeline de inferencia,
facilitando la reproducibilidad experimental y la gestión de hiperparámetros.
"""

from pathlib import Path
from typing import Dict, Any
import torch


class InferenceConfig:
    """
    Clase de configuración para el pipeline de inferencia.
    
    Attributes:
        BASE_DIR: Directorio raíz del proyecto.
        MODEL_ID: Identificador del modelo base de Stable Diffusion en HuggingFace.
        LORA_WEIGHTS_PATH: Ruta a los pesos LoRA entrenados.
        OUTPUT_DIR: Directorio de salida para imágenes generadas.
        DEVICE: Dispositivo de cómputo (cuda/cpu).
        DTYPE: Tipo de dato para tensores (float16 en GPU, float32 en CPU).
    """
    
    # Rutas del proyecto
    BASE_DIR = Path(__file__).parent.parent.resolve()
    MODEL_DIR = BASE_DIR / "models"
    LORA_WEIGHTS_PATH = MODEL_DIR / "lora_weights" / "pytorch_lora_weights.safetensors"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Configuración del modelo
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    
    # Configuración de hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    
    # Parámetros de generación por defecto
    DEFAULT_INFERENCE_PARAMS: Dict[str, Any] = {
        "num_inference_steps": 30,
        "guidance_scale": 7.0,
        "height": 512,
        "width": 512,
    }
    
    # Parámetros de LoRA
    LORA_ADAPTER_WEIGHTS = [1.0]  # Peso de influencia del LoRA (1.0 = máxima influencia)
    
    # Prompt optimizado para generación de plántulas de sorgo (nadir orthophoto)
    # Condensado para cumplir límite de 77 tokens de CLIP manteniendo características clave
    OPTIMIZED_SORGHUM_PROMPT = (
        "true nadir orthophoto, camera perpendicular to ground, "
        "single sorghum seedling centered, monocot grass, 4-7 narrow linear leaves, "
        "parallel venation, plant 20% of image, sharp focus, natural matte green, "
        "bare soil background, fine granular texture, diffuse daylight, "
        "realistic agronomy photo, neutral color, low saturation"
    )
    
    # Prompts alternativos plant-centric (enfoque en planta individual)
    DEFAULT_PROMPTS = [
        OPTIMIZED_SORGHUM_PROMPT,
        "realistic plant photo, single maize plant, centered subject, close-up crop, soil background, natural light, sharp focus, high detail",
        "realistic plant photo, single atriplex plant, centered subject, close-up crop, soil background, natural light, sharp focus, high detail",
        "macro plant photo, single seedling, centered subject, soil background, natural light, realistic texture, sharp focus"
    ]
    
    # Prompt base para generación (optimizado de sorghum)
    DEFAULT_PROMPT = OPTIMIZED_SORGHUM_PROMPT
    
    # Negative prompt detallado para evitar características no deseadas
    # Optimizado para cumplir límite de 77 tokens de CLIP
    NEGATIVE_PROMPT = (
        "aerial field, panorama, horizon, landscape, multiple plants, weeds, "
        "broadleaf, dicot, lobed leaf, reticulate venation, branches, flower, "
        "pot, studio, bokeh, straw, hay, wood grain, fabric, watermark, text"
    )
    
    # Seeds para reproducibilidad
    DEFAULT_SEED = 123
    DEFAULT_SEEDS = [123, 999]
    
    @classmethod
    def validate_paths(cls) -> None:
        """
        Valida que las rutas críticas del proyecto existan.
        
        Raises:
            FileNotFoundError: Si alguna ruta crítica no existe.
        """
        if not cls.LORA_WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Archivo de pesos LoRA no encontrado: {cls.LORA_WEIGHTS_PATH}"
            )
        
        # Crear directorio de salida si no existe
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_summary(cls) -> str:
        """
        Retorna un resumen de la configuración actual.
        
        Returns:
            String con información de configuración del sistema.
        """
        return (
            f"Configuración del Sistema\n"
            f"{'='*50}\n"
            f"Dispositivo: {cls.DEVICE}\n"
            f"Tipo de dato: {cls.DTYPE}\n"
            f"Modelo base: {cls.MODEL_ID}\n"
            f"Pesos LoRA: {cls.LORA_WEIGHTS_PATH.name}\n"
            f"Directorio de salida: {cls.OUTPUT_DIR}\n"
        )
