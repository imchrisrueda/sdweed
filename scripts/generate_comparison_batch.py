#!/usr/bin/env python3
"""
Script para generar comparaciones batch entre modelo base y LoRA.

Este script replica la funcionalidad del script SLURM original, generando
imágenes con y sin adaptación LoRA para múltiples prompts y seeds.

Uso:
    python scripts/generate_comparison_batch.py

Estructura de salida:
    outputs/
        base/
            base_p1_seed123.png
            base_p1_seed999.png
            ...
        lora/
            lora_p1_seed123.png
            lora_p1_seed999.png
            ...
"""

from pathlib import Path
import sys
import torch
from tqdm import tqdm

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import StableDiffusionInference
from config import InferenceConfig


def main():
    """Función principal de generación batch."""
    
    # Configurar directorios de salida
    base_dir = InferenceConfig.OUTPUT_DIR / "base"
    lora_dir = InferenceConfig.OUTPUT_DIR / "lora"
    base_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)
    
    # Prompts plant-centric (enfoque en planta individual)
    prompts = InferenceConfig.DEFAULT_PROMPTS
    
    # Seeds para generar múltiples variaciones
    seeds = InferenceConfig.DEFAULT_SEEDS
    
    # Configurar sistema
    print("="*70)
    print("GENERACIÓN BATCH: BASE vs LoRA")
    print("="*70)
    print(f"Dispositivo: {InferenceConfig.DEVICE}")
    print(f"Tipo de dato: {InferenceConfig.DTYPE}")
    print(f"Número de prompts: {len(prompts)}")
    print(f"Seeds por prompt: {seeds}")
    print(f"Total de imágenes a generar: {len(prompts) * len(seeds) * 2}")
    print("="*70)
    print()
    
    # Inicializar sistema de inferencia
    inference = StableDiffusionInference()
    
    # ========================================
    # GENERACIÓN BASE (sin LoRA)
    # ========================================
    print("📸 FASE 1: Generación con modelo BASE (sin LoRA)")
    print("-"*70)
    
    inference.initialize_pipeline(load_lora=False)
    
    total_base = len(prompts) * len(seeds)
    with tqdm(total=total_base, desc="Base", unit="img") as pbar:
        for i, prompt in enumerate(prompts, 1):
            for seed in seeds:
                # Generar imagen
                img = inference.generate_image(
                    prompt=prompt,
                    negative_prompt=InferenceConfig.NEGATIVE_PROMPT,
                    seed=seed,
                    num_inference_steps=30,
                    guidance_scale=7.0,
                    height=512,
                    width=512,
                )
                
                # Guardar imagen
                output_path = base_dir / f"base_p{i}_seed{seed}.png"
                img.save(output_path)
                
                pbar.update(1)
    
    # Limpiar memoria
    del inference.pipe
    import gc
    gc.collect()
    if InferenceConfig.DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    print(f"✅ Generadas {total_base} imágenes base")
    print(f"📁 Guardadas en: {base_dir}")
    print()
    
    # ========================================
    # GENERACIÓN LoRA (con adaptación)
    # ========================================
    print("📸 FASE 2: Generación con adaptación LoRA")
    print("-"*70)
    
    # Reinicializar con LoRA
    inference = StableDiffusionInference()
    inference.initialize_pipeline(load_lora=True)
    
    # Verificar peso del adaptador
    adapter_weight = InferenceConfig.LORA_ADAPTER_WEIGHTS[0]
    print(f"Peso del adaptador LoRA: {adapter_weight}")
    print()
    
    total_lora = len(prompts) * len(seeds)
    with tqdm(total=total_lora, desc="LoRA", unit="img") as pbar:
        for i, prompt in enumerate(prompts, 1):
            for seed in seeds:
                # Generar imagen con LoRA
                img = inference.generate_image(
                    prompt=prompt,
                    negative_prompt=InferenceConfig.NEGATIVE_PROMPT,
                    seed=seed,
                    num_inference_steps=30,
                    guidance_scale=7.0,
                    height=512,
                    width=512,
                )
                
                # Guardar imagen
                output_path = lora_dir / f"lora_p{i}_seed{seed}.png"
                img.save(output_path)
                
                pbar.update(1)
    
    print(f"✅ Generadas {total_lora} imágenes con LoRA")
    print(f"📁 Guardadas en: {lora_dir}")
    print()
    
    # ========================================
    # RESUMEN FINAL
    # ========================================
    print("="*70)
    print("✨ GENERACIÓN COMPLETADA")
    print("="*70)
    print(f"📂 BASE: {base_dir}")
    print(f"   └─ {total_base} imágenes")
    print(f"📂 LoRA: {lora_dir}")
    print(f"   └─ {total_lora} imágenes")
    print(f"📊 TOTAL: {total_base + total_lora} imágenes generadas")
    print()
    print("💡 Tip: Compara las imágenes con el mismo prompt y seed")
    print("   para evaluar el impacto de la adaptación LoRA.")
    print("="*70)


if __name__ == "__main__":
    main()
