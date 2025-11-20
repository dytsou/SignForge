# SignForge: A Modular Framework for Traffic Sign Recognition

A comprehensive, modular framework for traffic sign recognition with support for multiple architectures, parallel training, and advanced visualization capabilities.

## Project Structure

```
src/
├── __init__.py                 # Main package imports
├── README.md
│
├── core/                       # Core Functionality
│   ├── __init__.py
│   ├── dataset.py             # GTSRBDataset class
│   ├── models.py              # All model architectures & profiler
│   └── utils.py               # Training utilities & helpers
│
├── training/                   # Training Scripts  
│   ├── __init__.py
│   ├── train_baseline.py      # Baseline CNN training
│   └── train_mobilenet.py     # MobileNet training
│
├── ablation/                   # Ablation Study Framework
│   ├── __init__.py
│   ├── ablation_study.py      # Main ablation study orchestrator
│   ├── ablation_utils.py      # Training utilities for ablation
│   └── ablation_visualizer.py # Specialized visualizations
│
├── visualization/              # Visualization Tools
│   ├── __init__.py
│   ├── base_visualizer.py     # Core functionality and base class
│   ├── training_plots.py      # Training visualization plots
│   ├── classification_plots.py # Classification analysis plots
│   ├── comparison_plots.py    # Model comparison and ablation studies
│   ├── visualizer.py          # Main TrainingVisualizer class
│   ├── demo_visualization.py  # Demo & usage examples
│   ├── visualization.sh       # Shell script interface for report generation
│   └── README.md              # Visualization module documentation
│
├── analysis/                   # Analysis & Comparison
│   ├── __init__.py
│   └── compare_models.py      # Model comparison tools
│
└── data_prep/                  # Data Preparation
    ├── __init__.py
    └── prepare_data.py         # Data download & preprocessing
```

## Quick Start

### Basic Training
```python
# Train baseline model
python src/training/train_baseline.py

# Train MobileNet
python src/training/train_mobilenet.py
```

### Ablation Studies
```python
# Quick ablation study (3 models, 5 epochs)
python src/ablation/ablation_study.py --mode quick

# Full ablation study (8 models, 20 epochs, parallel)
python src/ablation/ablation_study.py --mode full --parallel

# Custom ablation study
python src/ablation/ablation_study.py --mode custom --models simple_cnn resnet18 --epochs 15
```

### Visualization Demo
```python
python src/visualization/demo_visualization.py
```

## Module Details

### Core Module (`src/core/`)
**Purpose**: Foundational components used across the framework
- **`dataset.py`**: GTSRBDataset with automatic augmentation
- **`models.py`**: 8+ model architectures + ModelProfiler for analysis
- **`utils.py`**: Training loops, evaluation, checkpointing, utilities

**Key Features**:
- Unified dataset interface with configurable transforms
- Comprehensive model profiling (parameters, size, inference time)
- Generic training functions supporting multiple optimizers
- GTSRB class names and evaluation utilities

### Training Module (`src/training/`)
**Purpose**: Individual model training scripts with visualization
- **`train_baseline.py`**: SimpleCNN training with profiling
- **`train_mobilenet.py`**: Transfer learning with MobileNetV2

**Key Features**:
- Real-time training visualization
- Automatic model profiling and checkpointing
- Comprehensive training reports with confusion matrices

### Ablation Module (`src/ablation/`)
**Purpose**: Comprehensive ablation study framework with parallel execution
- **`ablation_study.py`**: Main orchestrator with parallel/sequential modes
- **`ablation_utils.py`**: Parallel training utilities and configurations
- **`ablation_visualizer.py`**: 40+ specialized visualizations per study

**Key Features**:
- Parallel training across multiple GPUs/devices (2-4x speedup)
- 8+ model architectures with automatic profiling
- Individual model reports + comparative analysis
- HTML gallery generation for results presentation
- JSON export for programmatic access

### Visualization Module (`src/visualization/`)
**Purpose**: Advanced visualization and analysis tools with modular architecture
- **`base_visualizer.py`**: Core functionality and metric storage
- **`training_plots.py`**: Training curves, learning rate schedules, loss landscapes
- **`classification_plots.py`**: Confusion matrices, per-class metrics, classification reports
- **`comparison_plots.py`**: Model comparison, ablation studies, Pareto analysis
- **`visualizer.py`**: TrainingVisualizer unified interface using multiple inheritance
- **`demo_visualization.py`**: Usage examples and demonstrations
- **`visualization.sh`**: Shell script interface for command-line report generation

**Key Features**:
- **Modular Components**: Each visualization type in separate, focused files
- **Unified Interface**: TrainingVisualizer combines all functionality seamlessly
- **Command-Line Tools**: Shell script for batch processing and automation
- **Publication Quality**: 300 DPI figures with professional styling
- **Backward Compatibility**: Existing code continues to work unchanged
- **Interactive Reports**: HTML galleries with responsive design

### Analysis Module (`src/analysis/`)
**Purpose**: Model evaluation and comparison tools
- **`compare_models.py`**: Comprehensive model comparison and evaluation

**Key Features**:
- Automatic detection of trained models
- Performance benchmarking and recommendations
- Integration with ablation study results
- Production deployment guidance

### Data Preparation Module (`src/data_prep/`)
**Purpose**: Data preprocessing and preparation utilities
- **`prepare_data.py`**: GTSRB download, extraction, and processing

**Key Features**:
- Automatic download from official sources
- Format conversion (PPM → PNG)
- Train/test splitting with stratification
- Kaggle dataset support

## Usage Examples

### Using Core Components
```python
from src.core import GTSRBDataset, SimpleCNN, ModelProfiler
from src.visualization import TrainingVisualizer

# Create dataset and model
dataset = GTSRBDataset("data/annotations/train.json", ".", train=True)
model = SimpleCNN()

# Profile model characteristics
profile = ModelProfiler.get_model_profile(model)
print(f"Parameters: {profile['parameters_millions']:.2f}M")

# Setup visualization
visualizer = TrainingVisualizer(save_dir="results/my_experiment")
```

### Running Ablation Studies
```python
from src.ablation import run_ablation_study_parallel, quick_ablation_study

# Quick study for prototyping
results = quick_ablation_study(root_path=".", parallel=True)

# Full comprehensive study
results = run_ablation_study_parallel(
    models_to_test=['simple_cnn', 'resnet18', 'efficientnet_b0'],
    epochs=20,
    batch_size=32,
    save_models=True,
    max_parallel=4
)
```

### Advanced Visualization
```python
from src.visualization import TrainingVisualizer

visualizer = TrainingVisualizer(save_dir="results/analysis")

# Generate comprehensive ablation study plots
visualizer.plot_ablation_study(ablation_results)

# Create model comparison charts
visualizer.plot_model_comparison(model_metrics)

# Generate training reports
visualizer.generate_training_report(model, test_loader, device, "my_model", class_names)
```

### Command-Line Visualization
```bash
# Generate all visualizations for a model
./src/visualization/visualization.sh all -m mobilenet_v2 -s ./outputs

# Generate only training plots
./src/visualization/visualization.sh training-plots -m baseline_cnn

# Generate comprehensive HTML report
./src/visualization/visualization.sh report -m efficientnet -c "Stop,Yield,Speed_Limit"

# List all available visualization classes
./src/visualization/visualization.sh list-classes
```

## 🔧 Configuration Options

### Ablation Study Configuration
- **Models**: 8+ architectures (SimpleCNN, ResNet18, EfficientNet, VGG, etc.)
- **Parallelism**: Multi-GPU, MPS (Apple Silicon), CPU multiprocessing
- **Modes**: Quick (3 models, 5 epochs), Full (8 models, 20 epochs), Custom
- **Output**: JSON results, HTML galleries, publication-ready figures

### Visualization Options
- **Training Curves**: Loss, accuracy, learning rate, timing
- **Model Analysis**: Confusion matrices, per-class accuracy, profiling
- **Comparison Charts**: Bubble plots, ranking charts, efficiency analysis
- **Export Formats**: PNG (300 DPI), HTML galleries, JSON data

## Features

### Performance Benefits
- **Modular Design**: Easier maintenance and testing
- **Parallel Training**: 2-4x speedup on multi-GPU systems
- **Memory Efficiency**: Optimized data loading and model profiling
- **Comprehensive Logging**: Detailed metrics and error handling

### Visualization Capabilities
- **Modular Architecture**: Component-based design with focused responsibilities
- **Real-time Monitoring**: Live training curves and metrics
- **Publication Quality**: 300 DPI figures with professional styling
- **Interactive Galleries**: HTML reports with responsive design
- **Command-Line Interface**: Shell script for automation and batch processing
- **Comparative Analysis**: Multi-model ranking and efficiency analysis
- **Backward Compatibility**: Unified interface preserves existing workflows

### Research Tools
- **Ablation Studies**: Systematic architecture comparison
- **Model Profiling**: Automatic parameter counting and timing
- **Efficiency Analysis**: Accuracy/parameter trade-off studies
- **Deployment Guidance**: Production, mobile, and research recommendations

## Getting Started

1. **Prepare Data**: `python src/data_prep/prepare_data.py`
2. **Train Models**: `python src/training/train_baseline.py`
3. **Run Ablation**: `python src/ablation/ablation_study.py --mode quick`
4. **Generate Visualizations**: `./src/visualization/visualization.sh all -m your_model`
5. **Analyze Results**: `python src/analysis/compare_models.py`
6. **View Demo**: `python src/visualization/demo_visualization.py`

## Output Structure

Results are organized in the `results/` directory:
```
results/
├── visualizations/           # Individual model results
│   ├── baseline/             # Baseline model plots & metrics
│   └── mobilenet/            # MobileNet results
├── ablation/                 # Ablation study results
│   ├── simple_cnn/           # Per-model analysis
│   ├── resnet18/            
│   └── ...
├── comparison/                 # Comparative analysis
│   ├── bubble_analysis.png     # Efficiency scatter plots
│   ├── ranking_comparison.png  # Model rankings
│   └── ablation_study.png      # Main comparison plot
└── ablation_study_gallery.html # Interactive results browser
```

This modular structure makes the codebase more maintainable, testable, and suitable for both research and production use cases. 
