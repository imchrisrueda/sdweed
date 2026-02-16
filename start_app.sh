#!/bin/bash

# Script de inicio rápido para la aplicación web de generación de imágenes
# Autor: Sistema de Inferencia SD + LoRA
# Descripción: Inicializa y ejecuta la aplicación Streamlit con configuración óptima
# Versión: 2.0 (Plant-Centric Update)

echo "=============================================="
echo "  Generación de Imágenes Agrícolas - SD LoRA"
echo "  Versión 2.0 - Plant-Centric Update"
echo "=============================================="
echo ""
echo "✨ Nuevas Características:"
echo "  • 4 prompts plant-centric predefinidos"
echo "  • Negative prompts configurables"
echo "  • Peso LoRA ajustable (0.0 - 1.0)"
echo "  • Enfoque en plantas individuales"
echo ""

# Verificar que se está en el directorio correcto
if [ ! -f "app.py" ]; then
    echo "Error: app.py no encontrado. Ejecute este script desde el directorio raíz del proyecto."
    exit 1
fi

# Verificar instalación de Streamlit
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit no está instalado."
    echo "Instalando dependencias..."
    pip install -r requirements.txt
    echo ""
fi

# Verificar pesos LoRA
if [ ! -f "models/lora_weights/pytorch_lora_weights.safetensors" ]; then
    echo "Advertencia: Pesos LoRA no encontrados en models/lora_weights/"
    echo "Por favor, asegúrese de que los pesos estén en la ubicación correcta."
    echo ""
    read -p "¿Desea continuar de todos modos? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Información del sistema
echo "Verificando hardware..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU NVIDIA detectada:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "GPU no detectada. La aplicación se ejecutará en CPU."
    echo "Nota: La generación será considerablemente más lenta."
fi
echo ""

# Limpiar caché de Streamlit (opcional)
if [ "$1" == "--clear-cache" ]; then
    echo "Limpiando caché de Streamlit..."
    rm -rf .streamlit/cache
    echo "Caché limpiado."
    echo ""
fi

# Iniciar aplicación
echo "Iniciando aplicación web..."
echo "La aplicación se abrirá automáticamente en el navegador."
echo ""
echo "Acceso manual:"
echo "  - Local: http://localhost:8501"
echo "  - Red: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "💡 Otros scripts disponibles:"
echo "  • python scripts/generate_comparison_batch.py  (generación batch base vs LoRA)"
echo "  • python scripts/generate_images.py            (generación simple)"
echo "  • python scripts/test_inference.py             (testing del pipeline)"
echo ""
echo "Presione Ctrl+C para detener la aplicación."
echo "=============================================="
echo ""

# Ejecutar Streamlit
streamlit run app.py --server.headless false --server.address 0.0.0.0
