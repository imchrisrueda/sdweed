"""
Script principal para generación de imágenes sintéticas con Stable Diffusion + LoRA.

Este script implementa el flujo de trabajo de inferencia, generando imágenes
comparativas con y sin adaptación LoRA para evaluar el impacto del fine-tuning
específico de dominio en la calidad visual y coherencia estructural.
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import StableDiffusionInference, validate_cuda_availability
from config import InferenceConfig


def main():
    """
    Función principal de ejecución del pipeline de generación.
    
    Flujo de trabajo:
    1. Validación de entorno computacional (CUDA/CPU)
    2. Inicialización de configuración y pipeline
    3. Generación de imagen base (modelo SD sin adaptación)
    4. Generación de imagen con LoRA (modelo adaptado)
    5. Almacenamiento de resultados con nomenclatura estructurada
    """
    
    # Validación del entorno computacional
    # La disponibilidad de GPU reduce el tiempo de inferencia de ~5min a ~10s por imagen
    device_name, cuda_available = validate_cuda_availability()
    print(f"Dispositivo de cómputo: {device_name}")
    print(f"CUDA disponible: {cuda_available}")
    print()
    
    # Visualización de configuración del sistema
    print(InferenceConfig.get_summary())
    print()
    
    # Parámetros de generación
    # El prompt describe la vista aérea característica de sistemas UAS agrícolas
    # y especifica atributos morfológicos del cultivo objetivo (sorghum)
    prompt = InferenceConfig.DEFAULT_PROMPT
    seed = InferenceConfig.DEFAULT_SEED
    
    print("Iniciando generación de imágenes sintéticas...")
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")
    print()
    
    # Instanciación del sistema de inferencia
    inference = StableDiffusionInference(config=InferenceConfig)
    
    # Generación comparativa (base vs LoRA)
    # Esta comparación permite cuantificar la mejora en coherencia visual
    # introducida por el fine-tuning específico de dominio
    try:
        path_base, path_lora = inference.generate_comparison(
            prompt=prompt,
            seed=seed,
            output_prefix="output",
        )
        
        print("Generación completada exitosamente")
        print(f"Imagen base guardada en: {path_base}")
        print(f"Imagen LoRA guardada en: {path_lora}")
        
    except Exception as e:
        print(f"Error durante la generación: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
