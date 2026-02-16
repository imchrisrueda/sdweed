#!/usr/bin/env python3
"""
Script para verificar que los prompts cumplen el límite de 77 tokens de CLIP.

CLIP tokeniza el texto y tiene un límite máximo de 77 tokens por prompt.
Este script verifica que los prompts configurados no excedan este límite.

Uso:
    python scripts/verify_prompt_tokens.py
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import InferenceConfig

try:
    from transformers import CLIPTokenizer
    tokenizer_available = True
except ImportError:
    tokenizer_available = False
    print("⚠️  transformers no disponible. Usando conteo aproximado de palabras.")
    print("   Para conteo exacto: pip install transformers")
    print()


def count_tokens_exact(text: str) -> int:
    """Cuenta tokens usando el tokenizer de CLIP."""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokens = tokenizer.encode(text)
    return len(tokens)


def count_tokens_approx(text: str) -> int:
    """Aproximación: cuenta palabras y símbolos separados por espacios."""
    # CLIP usa BPE, pero como aproximación contamos palabras
    # Suele ser ligeramente mayor que el conteo real de tokens BPE
    words = text.replace(',', ' , ').replace('-', ' - ').split()
    return len(words)


def verify_prompt(name: str, prompt: str, max_tokens: int = 77):
    """Verifica que un prompt esté dentro del límite de tokens."""
    if tokenizer_available:
        token_count = count_tokens_exact(prompt)
        method = "CLIP tokenizer"
    else:
        token_count = count_tokens_approx(prompt)
        method = "aproximado"
    
    status = "✅" if token_count <= max_tokens else "❌"
    
    print(f"{status} {name}")
    print(f"   Tokens: {token_count}/{max_tokens} ({method})")
    
    if token_count > max_tokens:
        print(f"   ⚠️  EXCEDE el límite por {token_count - max_tokens} tokens")
        print(f"   El texto será truncado durante la inferencia")
    
    print()
    
    return token_count <= max_tokens


def main():
    """Función principal."""
    print("="*70)
    print("VERIFICACIÓN DE TOKENS EN PROMPTS")
    print("="*70)
    print(f"Límite CLIP: 77 tokens máximo por prompt")
    print()
    
    all_valid = True
    
    # Verificar prompt principal optimizado
    all_valid &= verify_prompt(
        "Prompt Principal (Sorghum Orthophoto)",
        InferenceConfig.OPTIMIZED_SORGHUM_PROMPT
    )
    
    # Verificar negative prompt
    all_valid &= verify_prompt(
        "Negative Prompt",
        InferenceConfig.NEGATIVE_PROMPT
    )
    
    # Verificar prompts alternativos
    for i, prompt in enumerate(InferenceConfig.DEFAULT_PROMPTS, 1):
        all_valid &= verify_prompt(
            f"Preset {i}",
            prompt
        )
    
    print("="*70)
    if all_valid:
        print("✅ TODOS LOS PROMPTS CUMPLEN EL LÍMITE DE 77 TOKENS")
    else:
        print("❌ ALGUNOS PROMPTS EXCEDEN EL LÍMITE")
        print("   Recomendación: Acortar prompts problemáticos")
    print("="*70)
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
