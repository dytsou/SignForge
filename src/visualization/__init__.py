"""
Visualization tools for training analysis and model comparison

This module provides comprehensive visualization capabilities for deep learning workflows:

- TrainingVisualizer: Unified interface for all visualization functionality
- TrainingPlots: Training curves, learning rate schedules, gradient analysis
- ClassificationPlots: Confusion matrices, per-class metrics, classification reports
- ComparisonPlots: Model comparison, ablation studies, Pareto analysis
- BaseVisualizer: Core functionality and shared utilities

Note: Report generation functionality is now available via visualization.sh script

Example usage:
    from visualization import TrainingVisualizer
    
    viz = TrainingVisualizer(save_dir="results/my_model")
    viz.log_metrics(epoch=1, train_loss=0.5, val_accuracy=85.0)
    viz.plot_training_curves()
"""

from .visualizer import TrainingVisualizer
from .training_plots import TrainingPlots
from .classification_plots import ClassificationPlots
from .comparison_plots import ComparisonPlots
from .base_visualizer import BaseVisualizer

__all__ = [
    'TrainingVisualizer',      # Main unified interface
    'TrainingPlots',           # Training-specific plots
    'ClassificationPlots',     # Classification analysis
    'ComparisonPlots',         # Model comparison and ablation
    'BaseVisualizer'           # Base functionality
] 