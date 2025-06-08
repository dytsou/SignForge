"""
Traffic Recognition - Modular Deep Learning Framework

This package provides a comprehensive framework for traffic sign recognition
with support for multiple model architectures, ablation studies, and 
advanced visualization capabilities.

Structure:
- core/         : Core functionality (models, dataset, utilities)
- training/     : Training scripts for different architectures  
- ablation/     : Comprehensive ablation study framework
- visualization/: Advanced visualization and analysis tools
- analysis/     : Model comparison and evaluation tools
- data_prep/    : Data preparation and preprocessing utilities
"""

# Core imports for easy access
from core import GTSRBDataset, SimpleCNN, LightCNN, DeepCNN, WideCNN
from core import ModelProfiler, get_model, seed_everything
from visualization import TrainingVisualizer
from ablation import run_ablation_study_parallel, quick_ablation_study, full_ablation_study

__version__ = "2.0.0"
__author__ = "Traffic Sign Recognition Team"

__all__ = [
    'GTSRBDataset', 'SimpleCNN', 'LightCNN', 'DeepCNN', 'WideCNN',
    'ModelProfiler', 'get_model', 'seed_everything', 'TrainingVisualizer',
    'run_ablation_study_parallel', 'quick_ablation_study', 'full_ablation_study'
] 