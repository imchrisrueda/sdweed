"""
Script de prueba para validación de la inferencia con LoRA.

Este script verifica la correcta configuración del entorno, la accesibilidad
de los pesos LoRA y la funcionalidad del pipeline de generación mediante
una ejecución simplificada de prueba.
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import StableDiffusionInference, validate_cuda_availability
from config import InferenceConfig


def main():
    """
    Función de validación del sistema de inferencia.
    
    Ejecuta verificaciones de:
    - Disponibilidad de hardware de aceleración
    - Existencia de archivos de pesos LoRA
    - Funcionalidad del pipeline de inferencia
    - Capacidad de escritura en directorio de salida
    """
    
    print("Iniciando prueba del sistema de inferencia")
    print()
    
    # Validación de entorno
    device_name, cuda_available = validate_cuda_availability()
    print(f"Dispositivo: {device_name}")
    print(f"CUDA disponible: {cuda_available}")
    print()
    
    # Validación de configuración
    try:
        InferenceConfig.validate_paths()
        print(f"Validación de rutas: OK")
        print(f"Pesos LoRA encontrados en: {InferenceConfig.LORA_WEIGHTS_PATH}")
        print()
    except FileNotFoundError as e:
        print(f"Error de validación: {e}")
        sys.exit(1)
    
    # Inicialización del sistema de inferencia
    print("Inicializando pipeline de inferencia...")
    inference = StableDiffusionInference(config=InferenceConfig)
    
    # Prueba de generación con LoRA
    # Se utiliza un número reducido de pasos para acelerar la validación
    print("Generando imagen de prueba con LoRA...")
    inference.initialize_pipeline(load_lora=True)
    
    prompt = InferenceConfig.DEFAULT_PROMPT
    seed = InferenceConfig.DEFAULT_SEED
    
    # Generación con configuración reducida para prueba rápida
    image = inference.generate_image(
        prompt=prompt,
        seed=seed,
        num_inference_steps=25,  # Reducido para prueba rápida
    )
    
    # Almacenamiento de resultado
    output_path = InferenceConfig.OUTPUT_DIR / "test_lora.png"
    image.save(output_path)
    
    print(f"Prueba completada exitosamente")
    print(f"Imagen de prueba guardada en: {output_path}")
    print()
    print("El sistema está configurado correctamente y listo para inferencia")


if __name__ == "__main__":
    main()
