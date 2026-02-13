# 🚀 Quick Start - Aplicación Web

## Inicio Rápido en 3 Pasos

### 1️⃣ Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2️⃣ Iniciar la Aplicación
```bash
streamlit run app.py
```
**O utilizar el script automatizado:**
```bash
./start_app.sh
```

### 3️⃣ Generar Imágenes
La aplicación se abrirá en tu navegador. ¡Comienza a generar imágenes inmediatamente!

---

## Acceso Rápido

| Método | URL |
|--------|-----|
| **Local** | http://localhost:8501 |
| **Red Local** | http://\<tu-ip\>:8501 |

---

## Características Principales

### 🎨 Generación Simple
- Interfaz intuitiva
- Parámetros ajustables en tiempo real
- Descarga inmediata

### ⚖️ Comparación
- Visualización lado a lado
- Evaluación del impacto de LoRA
- Descarga independiente

### 📚 Galería
- Historial automático
- Trazabilidad completa
- Exportación de parámetros

### ℹ️ Información
- Documentación integrada
- Guías de parámetros
- Estado del sistema

---

## Parámetros Recomendados

| Parámetro | Valor Inicial | Rango Óptimo |
|-----------|---------------|--------------|
| **Guidance Scale** | 7.5 | 7.0 - 9.0 |
| **Pasos de Inferencia** | 30 | 30 - 50 |
| **Peso LoRA** | 0.3 | 0.3 - 0.5 |

---

## Solución Rápida de Problemas

### La aplicación no inicia
```bash
# Verificar instalación de Streamlit
pip install streamlit==1.31.0

# Verificar ubicación
cd sd_infer
streamlit run app.py
```

### Generación muy lenta
- ✅ Verificar si hay GPU disponible (sidebar)
- ⚠️ Reducir pasos de inferencia a 20-25
- 💡 Esperable en CPU: 5-10 minutos por imagen

### Error de memoria GPU
```bash
# Reiniciar la aplicación para limpiar caché
Ctrl + C
streamlit run app.py
```

---

## 📖 Documentación Completa

Para información detallada, consultar:
- **README.md**: Documentación principal del proyecto
- **docs/streamlit_app.md**: Guía completa de la aplicación
- **docs/usage.md**: Ejemplos de uso avanzado
- **docs/methodology.md**: Fundamentos técnicos

---

## Keyboard Shortcuts

| Atajo | Acción |
|-------|--------|
| `R` | Recargar aplicación |
| `M` | Abrir/cerrar menú |
| `C` | Limpiar caché |

---

## Tips de Uso

💡 **Tip 1**: Mantén el mismo seed para comparar diferentes configuraciones de parámetros

💡 **Tip 2**: Usa el historial para recuperar configuraciones exitosas

💡 **Tip 3**: Descarga las imágenes importantes inmediatamente (no persisten entre sesiones)

💡 **Tip 4**: Experimenta con diferentes pesos de LoRA para encontrar el balance óptimo

---

**¡Feliz generación de imágenes! 🌾**
