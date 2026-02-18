# Explorando Capas Convolucionales a Través de Datos y Experimentos

**Autor**: Deisy Lorena Guzmán Cabrales 

---

## Tabla de Contenidos
- [Descripción del Problema](#descripción-del-problema)
- [Descripción del Dataset](#descripción-del-dataset)
- [Diseño de Arquitectura](#diseño-de-arquitectura)
- [Resultados Experimentales](#resultados-experimentales)
- [Interpretación y Conocimientos Clave](#interpretación-y-conocimientos-clave)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Configuración y Requisitos](#configuración-y-requisitos)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Despliegue del Modelo](#despliegue-del-modelo)
- [Referencias](#referencias)

---

## Descripción del Problema

Este proyecto explora las **redes neuronales convolucionales (CNNs)** no como modelos de caja negra, sino como componentes arquitectónicos cuyas decisiones de diseño afectan directamente el rendimiento, escalabilidad e interpretabilidad.

### Objetivos de Aprendizaje

1. Comprender la intuición matemática detrás de las capas convolucionales
2. Analizar cómo las decisiones arquitectónicas (tamaño de kernel, profundidad, stride, padding) afectan el aprendizaje
3. Comparar capas convolucionales con capas totalmente conectadas para datos tipo imagen
4. Realizar análisis exploratorio de datos (EDA) significativo para tareas de redes neuronales
5. Comunicar decisiones arquitectónicas y experimentales de manera clara

### Enfoque

En lugar de seguir una receta, este proyecto:
- **Selecciona** un dataset apropiado con justificación
- **Analiza** características del dataset a través de EDA
- **Diseña** arquitecturas CNN desde cero con razonamiento explícito
- **Conduce** experimentos controlados sobre parámetros arquitectónicos
- **Interpreta** resultados a través del lente del sesgo inductivo
- **Despliega** el modelo para inferencia en producción

---

## Descripción del Dataset

### Fashion-MNIST

**Fuente**: [TensorFlow Keras Datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist)  
**Repositorio Original**: https://github.com/zalandoresearch/fashion-mnist

### Características del Dataset

- **Tarea**: Clasificación de imágenes multi-clase
- **Clases**: 10 categorías de moda
  - 0: Camiseta/top
  - 1: Pantalón
  - 2: Suéter
  - 3: Vestido
  - 4: Abrigo
  - 5: Sandalia
  - 6: Camisa
  - 7: Zapatilla deportiva
  - 8: Bolso
  - 9: Botín

- **Tamaño**: 
  - Conjunto de entrenamiento: 60,000 imágenes
  - Conjunto de prueba: 10,000 imágenes
- **Dimensiones de imagen**: 28×28 píxeles
- **Canales**: 1 (escala de grises)
- **Rango de píxeles**: [0, 255]
- **Balance de clases**: Perfectamente balanceado (6,000 muestras por clase en entrenamiento)

### ¿Por Qué Fashion-MNIST?

Fashion-MNIST es ideal para estudiar capas convolucionales porque:

1. **Estructura Espacial**: Las prendas de moda contienen patrones locales (texturas, bordes) que se benefician de la conectividad local
2. **Invariancia por Traslación**: Los objetos permanecen reconocibles independientemente de su posición – exactamente lo que proporciona la convolución
3. **Características Jerárquicas**: Las capas inferiores detectan bordes/texturas, las capas más profundas detectan partes de objetos (mangas, tacones)
4. **Desafiante pero Manejable**: Más difícil que los dígitos MNIST, haciendo visibles las diferencias arquitectónicas
5. **Tamaño Práctico**: Cabe en memoria y entrena rápidamente para experimentación rápida

---

## Diseño de Arquitectura

### Modelo Base (No Convolucional)

**Propósito**: Establecer referencia de rendimiento sin sesgo inductivo espacial

```
Input (784) 
    ↓
Dense(128, ReLU) + Dropout(0.2)
    ↓
Dense(64, ReLU) + Dropout(0.2)
    ↓
Dense(10, Softmax)
```

**Parámetros**: ~101,000  
**Limitación Clave**: Trata los píxeles como características independientes, ignorando la estructura espacial

---

### Modelo CNN (Convolucional)

**Arquitectura Propuesta**:

```
Input (28×28×1)
    ↓
Conv2D(32 filtros, 3×3, ReLU) + BatchNorm
    ↓
Conv2D(32 filtros, 3×3, ReLU) + BatchNorm
    ↓
MaxPooling(2×2)
    ↓
Conv2D(64 filtros, 3×3, ReLU) + BatchNorm
    ↓
Conv2D(64 filtros, 3×3, ReLU) + BatchNorm
    ↓
MaxPooling(2×2)
    ↓
Flatten
    ↓
Dense(128, ReLU) + Dropout(0.3)
    ↓
Dense(10, Softmax)
```

**Parámetros**: ~94,000  
**Ventaja**: Menos parámetros que el modelo base, mejor rendimiento

---

### Justificaciones Arquitectónicas

#### ¿Por Qué Kernels 3×3?
- **Eficiente**: Dos capas 3×3 = mismo campo receptivo que 5×5, pero menos parámetros
- **Más no-linealidad**: Apilar capas pequeñas agrega funciones de activación
- **Estándar de la industria**: Usado en VGG, ResNet y arquitecturas modernas

#### ¿Por Qué Dos Capas Conv Antes del Pooling?
- **Características jerárquicas**: La primera capa encuentra bordes, la segunda los combina
- **Preserva resolución**: No reduce el muestreo demasiado pronto
- **Agrega profundidad**: Más poder representacional

#### ¿Por Qué MaxPooling 2×2?
- **Reduce gradualmente**: Dimensiones espaciales 28→14→7
- **Invariancia por traslación**: Pequeños desplazamientos no afectan la salida
- **Reduce parámetros**: Hace que las capas más profundas sean computacionalmente factibles
- **Regularización**: Ayuda a prevenir el sobreajuste

#### ¿Por Qué Aumentar Filtros (32→64)?
- **Compensar pérdida espacial**: A medida que las dimensiones se reducen, agregar más canales
- **Capturar complejidad**: Las capas más profundas necesitan características más abstractas
- **Práctica estándar**: Común en arquitecturas CNN

#### ¿Por Qué BatchNormalization?
- **Estabiliza el entrenamiento**: Normaliza las entradas de las capas
- **Convergencia más rápida**: Permite tasas de aprendizaje más altas
- **Regularización**: Prevención leve del sobreajuste

---

## Resultados Experimentales

### Modelo Base vs CNN

| Modelo | Precisión de Prueba | Pérdida de Prueba | Parámetros | Tiempo de Entrenamiento |
|--------|---------------------|-------------------|------------|-------------------------|
| **Base (Dense)** | ~87-88% | ~0.35 | 101,000 | ~30s |
| **CNN (3×3)** | ~91-92% | ~0.25 | 94,000 | ~45s |
| **Mejora** | **+4-5%** | **-0.10** | **-7%** | +15s |

**Conclusión Clave**: CNN logra mejor precisión con menos parámetros al explotar la estructura espacial.

---

### Experimento Controlado: Tamaño de Kernel

**Pregunta de Investigación**: ¿Cómo afecta el tamaño del kernel el rendimiento del modelo?

**Variables de control**: Número de capas, filtros, pooling, hiperparámetros de entrenamiento

| Tamaño de Kernel | Precisión de Prueba | Parámetros | Tiempo de Entrenamiento | Observaciones |
|------------------|---------------------|------------|-------------------------|---------------|
| **3×3** | ~91.5% | 94,000 | 45s | Mejor balance entre precisión y eficiencia |
| **5×5** | ~91.3% | 250,000 | 65s | Más parámetros, precisión similar |
| **7×7** | ~89-90% | 450,000 | 85s | Demasiado agresivo para imágenes 28×28 |

**Conclusión**: Los kernels 3×3 son óptimos para Fashion-MNIST – kernels más grandes no mejoran la precisión pero aumentan la complejidad.

---

### Visualizaciones

El notebook incluye:
- Análisis de distribución de clases
- Imágenes de muestra por clase
- Distribuciones de intensidad de píxeles
- Curvas de entrenamiento/validación
- Gráficos de comparación de tamaño de kernel
- Análisis de compensaciones de rendimiento

---

## Interpretación y Conocimientos Clave

### Por Qué las CNNs Superan al Modelo Base

1. **Conectividad Local**: Aprovecha la correlación entre píxeles cercanos (bordes, texturas)
2. **Compartición de Parámetros**: El mismo filtro aplicado en todas partes → invariancia por traslación
3. **Aprendizaje Jerárquico**: Abstracción progresiva desde bordes hasta partes hasta objetos
4. **Alineación del Sesgo Inductivo**: Las suposiciones de CNN coinciden con la estructura de imagen

### ¿Qué Sesgo Inductivo Introduce la Convolución?

**Tres sesgos clave:**

1. **Localidad**: Los píxeles cercanos son más relevantes que los píxeles distantes
2. **Equivariancia por Traslación**: Los patrones son significativos independientemente de su posición
3. **Composición Jerárquica**: Los patrones complejos se construyen a partir de patrones más simples

Estos sesgos:
- Reducen el espacio de hipótesis → aprendizaje más rápido
- Requieren menos datos → mejor generalización
- Codifican conocimiento del dominio → rendimiento mejorado

### ¿Cuándo NO Sería Apropiada la Convolución?

La convolución es **inapropiada** para:

1. **Datos Tabulares/Estructurados**: Sin estructura espacial (edad, ingresos, etc.)
2. **Tareas Sensibles a la Posición**: La ubicación importa (diagnóstico en imagen médica)
3. **Dependencias de Largo Alcance**: Patrones alejados en el espacio
4. **Grafos Irregulares**: Redes sociales, moléculas (necesita GNNs)
5. **Datos Secuenciales con Orden Variable**: Algunas tareas de NLP (usar Transformers)
6. **Datasets Muy Pequeños**: Datos insuficientes para aprender filtros

**Conocimiento Clave**: Las elecciones arquitectónicas codifican suposiciones – las CNNs tienen éxito cuando las suposiciones coinciden con la estructura del problema.

---

## Estructura del Repositorio

```
Exploring-Convolutional-Layers-Through-Data-and-Experiments/
│
├── README.md                                    # Documentación del proyecto
├── requirements.txt                             # Dependencias de Python (PyTorch)
├── .gitignore                                   # Reglas de ignorar de Git
├── convolutional_layers_workshop.ipynb          # Notebook completo
│
└── fashion_mnist_cnn_model_pytorch/             # Artefactos del modelo entrenado 
```

---

## 🛠️ Configuración y Requisitos

### Prerrequisitos

- Python 3.9+ (¡Compatible con Python 3.14!)
- PyTorch 2.0+
- Jupyter Notebook o JupyterLab

### Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tuusuario/Exploring-Convolutional-Layers-Through-Data-and-Experiments.git
   cd Exploring-Convolutional-Layers-Through-Data-and-Experiments
   ```

2. **Crear entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install torch torchvision numpy matplotlib seaborn pandas jupyter
   ```

   O usar requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
matplotlib>=3.5.0
seaborn>=0.12.0
pandas>=1.5.0
jupyter>=1.0.0
onnx>=1.14.0  # Para exportación de modelos
```

---

## Cómo Ejecutar

### Opción 1: Jupyter Notebook

1. Iniciar Jupyter:
   ```bash
   jupyter notebook
   ```

2. Abrir `convolutional_layers_workshop.ipynb`

3. Ejecutar todas las celdas secuencialmente (Cell → Run All)

### Opción 2: JupyterLab

1. Iniciar JupyterLab:
   ```bash
   jupyter lab
   ```

2. Abrir `convolutional_layers_workshop.ipynb`

3. Ejecutar celdas en orden


El notebook:
- Descargará Fashion-MNIST automáticamente (solo en la primera ejecución)
- Generará visualizaciones en línea
- Mostrará el progreso de entrenamiento y métricas
- Guardará el modelo entrenado en `fashion_mnist_cnn_model_pytorch/`
- Usará GPU automáticamente si está disponible (CUDA)

---



## Conclusiones Clave

1. **Las redes neuronales no son cajas negras** – las elecciones arquitectónicas importan
2. **El sesgo inductivo es una característica, no un error** – codifica conocimiento del dominio
3. **La experimentación supera la intuición** – prueba las suposiciones sistemáticamente
4. **La simplicidad a menudo gana** – los kernels 3×3 superan alternativas más grandes
5. **Comprensión > Precisión** – saber POR QUÉ funciona tu modelo
