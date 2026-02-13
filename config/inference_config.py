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
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
    }
    
    # Parámetros de LoRA
    LORA_ADAPTER_WEIGHTS = [0.3]  # Peso de influencia del LoRA (rango típico: 0.1-0.6)
    
    # Prompt base para generación
    DEFAULT_PROMPT = (
        "aerial drone photograph, top-down view, agricultural field, "
        "natural soil texture, realistic, sorghum plant, early growth stage"
    )
    
    # Seed para reproducibilidad
    DEFAULT_SEED = 1234
    
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
