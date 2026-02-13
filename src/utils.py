"""
Utilidades auxiliares para el proyecto de inferencia con Stable Diffusion.

Este módulo proporciona funciones de soporte para validación, visualización
y gestión de archivos durante el proceso de generación de imágenes sintéticas.
"""

from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image


def validate_cuda_availability() -> Tuple[str, bool]:
    """
    Verifica la disponibilidad de CUDA y retorna información del dispositivo.
    
    La aceleración por GPU es crítica para la eficiencia computacional del
    pipeline de difusión, reduciendo el tiempo de inferencia significativamente.
    
    Returns:
        Tupla con (nombre_dispositivo, disponibilidad_cuda).
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return device_name, cuda_available


def create_output_directory(output_dir: Path) -> None:
    """
    Crea el directorio de salida si no existe.
    
    Args:
        output_dir: Ruta del directorio a crear.
    """
    output_dir.mkdir(parents=True, exist_ok=True)


def save_image_with_metadata(
    image: Image.Image,
    output_path: Path,
    metadata: dict
) -> None:
    """
    Guarda una imagen con metadatos embebidos en formato PNG.
    
    Los metadatos permiten trazabilidad experimental, registrando parámetros
    de generación como prompt, seed y configuración del modelo.
    
    Args:
        image: Imagen PIL a guardar.
        output_path: Ruta de destino.
        metadata: Diccionario con información a embeber.
    """
    from PIL.PngImagePlugin import PngInfo
    
    png_info = PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    
    image.save(output_path, pnginfo=png_info)


def list_generated_images(output_dir: Path) -> List[Path]:
    """
    Lista todas las imágenes generadas en el directorio de salida.
    
    Args:
        output_dir: Directorio donde se almacenan las imágenes.
    
    Returns:
        Lista de rutas a archivos de imagen (PNG).
    """
    return sorted(output_dir.glob("*.png"))


def format_experiment_filename(
    base_name: str,
    seed: int,
    lora_weight: float,
    extension: str = "png"
) -> str:
    """
    Genera un nombre de archivo estructurado para resultados experimentales.
    
    El esquema de nomenclatura permite identificación rápida de parámetros
    experimentales directamente desde el nombre del archivo.
    
    Args:
        base_name: Nombre base descriptivo.
        seed: Semilla utilizada en la generación.
        lora_weight: Peso del adaptador LoRA.
        extension: Extensión del archivo.
    
    Returns:
        String con nombre de archivo formateado.
    """
    return f"{base_name}_seed{seed}_lora{lora_weight:.2f}.{extension}"
