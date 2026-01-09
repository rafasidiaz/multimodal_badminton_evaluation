# Clasificación y evaluación de modelos de clasificación en bádminton mediante redes neuronales sobre un conjunto multimodal

Trabajo de Fin de Máster: Reconocimiento de patrones aplicado al deporte (Caso de Uso: Bádminton)

## Descripción del Proyecto

Este proyecto evalúa sistemáticamente cinco arquitecturas de redes neuronales (LSTM, ConvLSTM, Transformer, ParallelConvLSTM y ResNetLSTM) para clasificar 2 tipos de golpes de bádminton:
- Forehand Clear
- Backhand Clear

Sobre la base de múltiples configuraciones de sensores multimodales del dataset MultiSenseBadminton con 18 jugadores.

## Requisitos del Sistema

### Hardware
- GPU NVIDIA con CUDA (recomendado)
- Mínimo 16GB RAM
- 10GB espacio en disco

### Software
```bash
Python >= 3.8
CUDA >= 11.0 (para GPU)
```

### Dependencias principales
```bash
torch >= 1.10.0
h5py >= 3.0.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
optuna >= 3.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
tqdm >= 4.62.0
```

## Instalación

### 1. Clonar el repositorio
```bash
git clone <URL_REPOSITORIO>
cd tfm-badminton-classification
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install h5py numpy pandas scikit-learn optuna matplotlib seaborn scipy tqdm
```

### 4. Descargar los datos

El dataset original que se utiliza en el notebook 1.Preprocesado_de_los_datos se puede descargar en  https://doi.org/10.6084/m9.figshare.c.6725706.v1

La versión preprocesada del dataset que se utiliza en el notebook 2.entrenamiento_modelos_stroke_type.ipynb se puede descargar en https://doi.org/10.5281/zenodo.18202490


### 5. Configurar Google Colab (opcional)

El proyecto se ha llevado a cabo utilizando Google Colab, consideramos conveniente utilizar esta o algun otra solución alojada en la nube debido a la alta carga computacional.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 6. Limpieza y adaptación del código

La estructura del código se ha mantenido intacta para la evaluación del mismo, para utilizar el código correctamente, requiere modificaciones en varias rutas hard-coded

## Estructura del Proyecto

```
├── utils/
│   ├── dict_utils.py          # Utilidades para diccionarios
│   ├── preprocessing.py       # Dataset de PyTorch
│   ├── SciDataModels.py       # Modelos base (LSTM, ConvLSTM, Transformer)
│   ├── improved_models.py     # Modelos mejorados (ParallelConvLSTM)
│   └── alternative_models.py  # Modelos alternativos (ResNetLSTM)
├── results/                   # Resultados de experimentos
├── 1_Preprocesado_de_los_datos.ipynb
├── 2_entrenamiento_modelos_stroke_type.ipynb
└── README.md
```

## Configuración de Datos

### Dataset MultiSenseBadminton

El proyecto utiliza el dataset MultiSenseBadminton con las siguientes modalidades:

| Sensor | Streams | Canales | Descripción |
|--------|---------|---------|-------------|
| Eye Gaze | gaze | 2 | Seguimiento ocular |
| GForce EMG (Antebrazo) | emg-values | 8 | Electromiografía |
| GForce EMG (Brazo superior) | emg-values | 8 | Electromiografía |
| CGX AIM EMG (Pierna) | emg-values | 4 | Electromiografía |
| Moticon Insole | pressure/cop | 36 | Presión plantar |
| Perception Neuron Studio | Euler-angle | 63 | Movimiento corporal |

**Total: 121 canales**

El dataset original se puede descargar en  https://doi.org/10.6084/m9.figshare.c.6725706.v1
Una versión preprocesada del  dataset se puede descargar en https://doi.org/10.5281/zenodo.18202490

### Configuraciones de sensores disponibles

```python
sensor_subsets = [
    'allStreams',      # Todos los sensores (121 canales)
    'onlyBody',        # Solo Perception Neuron (63 canales)
    'onlyEye',         # Solo Eye Gaze (2 canales)
    'onlyGforce',      # Solo EMG brazos (16 canales)
    'onlyInsole',      # Solo plantillas (36 canales)
    'onlyCognionics'   # Solo EMG piernas (4 canales)
]
```

## Resultados

### Estructura de salidas

```
results/
├── hpo_results/
│   ├── {modelo}_{sensor}/
│   │   ├── best_params.json          # Mejores hiperparámetros
│   │   └── optuna_study.db          # Base de datos de búsqueda
│   └── summary_hpo.csv              # Resumen global HPO
└── training/
    ├── {modelo}_{sensor}/
    │   ├── fold_metrics/
    │   │   ├── {modelo}_folds_results.csv
    │   │   └── {modelo}_summary.json
    │   ├── LTO_{fold}_{test_subj}_{val_subj}/
    │   │   ├── model_S_{fold}        # Pesos del modelo
    │   │   ├── config_S_{fold}       # Configuración
    │   │   ├── loss.png              # Curvas de pérdida
    │   │   ├── accuracy.png          # Curvas de precisión
    │   │   ├── bal_acc.png           # Balanced accuracy
    │   │   ├── f1_acc.png            # F1-score
    │   │   ├── train_acc_confusion.png
    │   │   └── test_acc_confusion.png
    │   └── leave_three_out_splits.json
    └── summary_training.csv          # Resumen global entrenamiento
```

### Métricas evaluadas

- **Accuracy**: Precisión general
- **Balanced Accuracy**: Precisión balanceada entre clases
- **F1-Score**: Media armónica de precisión y recall (weighted)
- **Confusion Matrix**: Matriz de confusión normalizada

## Modelos Implementados

### 1. LSTM Baseline
Red LSTM de 2 capas con capas fully connected.
```python
class LSTMRefinedModel(nn.Module):
    # 64 unidades ocultas, 2 capas
```

### 2. ConvLSTM
Combinación de convoluciones 1D con LSTM y residual connections.
```python
class ConvLSTMRefinedModel(nn.Module):
    # Conv1D + BatchNorm + MaxPool + LSTM
```

### 3. Transformer
Arquitectura basada en self-attention con multi-head attention.
```python
class TransformerRefinedModel(nn.Module):
    # Embedding + TransformerBlock + FC
```

### 4. ParallelConvLSTM (Improved)
Arquitectura paralela con attention mechanism y squeeze-excitation.
```python
class ImprovedConvLSTMModel(nn.Module):
    # Parallel convolutions + Attention + SE blocks
```

### 5. ResNetLSTM
Inspirado en ResNet con bloques residuales y LSTM.
```python
class ResNetLSTMModel(nn.Module):
    # Residual blocks + LSTM + Global pooling
```

## Personalización

### Añadir nueva configuración de sensores

```python
# En archivo de configuración
elif subset == 'customConfig':
    # Definir índices de canales deseados
    input_feature_matrices = feature_matrices_current[:, :, canal_inicio:canal_fin]
```

### Añadir nuevo modelo

1. Crear clase en `utils/custom_models.py`:
```python
class NuevoModelo(nn.Module):
    def __init__(self, input_channels, hidden_features, output_size):
        super(NuevoModelo, self).__init__()
        # Definir capas
    
    def forward(self, x):
        # Implementar forward pass
        return x
```

2. Integrar en función de entrenamiento:
```python
elif modelName == "NuevoModelo":
    model = NuevoModelo(
        len(input_feature_matrices[0,0,:]),
        hidden_features,
        label_num
    ).to(device)
```

3. Definir espacio de hiperparámetros en función objetivo.

## Troubleshooting

### Error: CUDA out of memory
```python
# Reducir batch size
batch_size = 16  # o 8
```

### Error: HDF5 file not found
```bash
# Verificar ruta del archivo
ls data_processed/
# Ejecutar preprocesamiento primero
```

### Warning: Convergence issues
```python
# Aumentar paciencia o epochs
PATIENCE_TRAIN = 20
NUM_EPOCHS_TRAIN = 150
```

### Resultados inconsistentes
```python
# Fijar semilla aleatoria
set_random_seed(42)
```

## Recomendaciones

### Para experimentos rápidos
```python
# Configuración rápida
NUM_EPOCHS_OPTUNA = 10
FOLDS_OPTUNA = 2
model_list = ['LSTM']
sensor_subset_list = ['onlyBody']
```

### Para resultados publicables
```python
# Configuración completa
NUM_EPOCHS_OPTUNA = 50
FOLDS_OPTUNA = 5
model_list = ['LSTM', 'ConvLSTM', 'Transformer', 
              'ImprovedConvLSTMModel', 'ResNetLSTMModel']
sensor_subset_list = ['allStreams', 'onlyBody', 'onlyEye', 
                      'onlyGforce', 'onlyInsole', 'onlyCognionics']
```

## Contribuciones

Este código está basado en:
- [ActionNet](https://action-net.csail.mit.edu) para preprocesamiento de datos
- [MultiSenseBadminton](https://github.com/ywu840/MultiSenseBadminton) para dataset original

## Licencia

MIT License - Ver archivo LICENSE para más detalles.

## Contacto

Para preguntas sobre el código o metodología:
- Autor: Rafael Asián
- Email: [rasiand@uoc.edu]
- Institución: [Universidad Oberta de Catalunya]

## Citación

Si utilizas este código, por favor cita:

```bibtex
@mastersthesis{asian2025badminton,
    title={Clasificación de Técnicas de Bádminton mediante Redes Neuronales},
    author={Asián, Rafael},
    year={2025},
    school={Universidad},
    type={Master's Thesis}
}
```

## Referencias

1. Seong, M., et al. (2023). MultiSenseBadminton Dataset
2. DelPreto, J., et al. (2022). ActionNet: Multimodal Activity Recognition
3. Vaswani, A., et al. (2017). Attention Is All You Need
4. He, K., et al. (2016). Deep Residual Learning for Image Recognition