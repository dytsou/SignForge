"""
Utility functions for ablation study
"""

import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.dataset import GTSRBDataset
from core import utils
from visualization.visualizer import TrainingVisualizer
from core.models import get_model, ModelProfiler

def train_single_model(model_config, shared_data, gpu_device_id=None):
    """Train a single model - designed to be called in parallel"""
    model_name, epochs, batch_size, lr, optimizer_type, save_models = model_config
    root_path, train_json_path, test_json_path = shared_data
    
    try:
        # Set up device (distribute across available GPUs if multiple)
        if gpu_device_id is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_device_id}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        print(f"Starting {model_name} on {device}")
        
        # Setup data loaders for this worker  
        train_json_path = root_path / "data/annotations/train.json"
        test_json_path = root_path / "data/annotations/test.json"
        train_ds = GTSRBDataset(train_json_path, root_path, True)
        test_ds = GTSRBDataset(test_json_path, root_path, False)
        train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=2)
        
        # Create model
        model = get_model(model_name)
        model = model.to(device)
        
        # Profile model characteristics
        model_profile = ModelProfiler.get_model_profile(model, device=device)
        
        print(f"Model Profile:")
        print(f"   Parameters: {model_profile['parameters_millions']:.2f}M")
        print(f"   Model Size: {model_profile['model_size_mb']:.2f} MB")
        print(f"   Device: {device}")
        
        # Setup visualizer with model-specific directory
        viz_dir = root_path/f"results/ablation/{model_name}"
        visualizer = TrainingVisualizer(save_dir=viz_dir)
        
        # Train model
        start_time = time.time()
        model, visualizer = utils.train_model_generic(
            model, train_loader, test_loader, device, model_name,
            epochs=epochs, lr=lr, optimizer_type=optimizer_type,
            visualizer=visualizer, model_profile=model_profile
        )
        total_training_time = time.time() - start_time
        
        # Final evaluation
        final_acc = utils.evaluate(model, test_loader, device, f"[{model_name} Final]")
        
        # Save model if requested
        if save_models:
            model_path = root_path / f"models/ablation_{model_name}.pth"
            utils.save_checkpoint(model, model_path)
            print(f"Model saved to {model_path}")
        
        # Generate comprehensive training report with visualizations
        class_names = utils.get_class_names()
        visualizer.generate_training_report(model, test_loader, device, model_name, class_names)
        
        # Save additional model-specific visualizations
        from ablation.ablation_visualizer import save_individual_visualizations
        save_individual_visualizations(model, test_loader, device, model_name, 
                                     visualizer, model_profile, viz_dir)
        
        # Prepare results
        result = {
            'model_name': model_name,
            'accuracy': final_acc,
            'total_parameters': model_profile['total_parameters'],
            'trainable_parameters': model_profile['trainable_parameters'],
            'parameters_millions': model_profile['parameters_millions'],
            'model_size_mb': model_profile['model_size_mb'],
            'inference_time_ms': model_profile['inference_time_ms'],
            'total_training_time': total_training_time,
            'avg_epoch_time': total_training_time / epochs,
            'epochs': epochs,
            'learning_rate': lr,
            'optimizer': optimizer_type,
            'final_loss': visualizer.train_losses[-1] if visualizer.train_losses else 0,
            'best_accuracy': max(visualizer.val_accuracies) if visualizer.val_accuracies else final_acc,
            'device': str(device),
            'visualization_dir': str(viz_dir)
        }
        
        print(f"{model_name} completed - Accuracy: {final_acc:.2f}% on {device}")
        return result
        
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        return {'model_name': model_name, 'error': str(e)}

def get_training_config(model_name):
    """Get training configuration for a model"""
    if 'resnet' in model_name or 'vgg' in model_name or 'efficientnet' in model_name:
        lr = 1e-4 if 'scratch' not in model_name else 5e-4
        optimizer_type = 'adamw'
    else:
        lr = 1e-3
        optimizer_type = 'adam'
    return lr, optimizer_type

def determine_parallelism(max_parallel, use_gpu, num_models):
    """Determine optimal parallelism based on hardware"""
    if use_gpu and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        max_parallel = min(max_parallel, gpu_count, num_models)
        print(f"Using {gpu_count} GPU(s), running {max_parallel} models in parallel")
    elif torch.backends.mps.is_available():
        max_parallel = min(max_parallel, 2, num_models)  # MPS limitation
        print(f"Using MPS (Apple Silicon), running {max_parallel} models in parallel")
    else:
        import multiprocessing as mp
        cpu_count = mp.cpu_count()
        max_parallel = min(max_parallel, cpu_count // 2, num_models)
        print(f"Using CPU with {max_parallel} parallel processes")
    
    return max_parallel

def get_default_models():
    """Get default models for ablation study"""
    return [
        'light_cnn',      # Ultra-lightweight
        'simple_cnn',     # Baseline
        'wide_cnn',       # Wider baseline
        'deep_cnn',       # Deeper baseline
        'resnet18',       # Standard architecture (pretrained)
        'resnet18_scratch', # Standard architecture (from scratch)
        'efficientnet_b0', # Efficient architecture
        'vgg11',          # Classical architecture
    ]

def prepare_model_configs(models_to_test, epochs, batch_size, save_models):
    """Prepare training configurations for all models"""
    model_configs = []
    for model_name in models_to_test:
        lr, optimizer_type = get_training_config(model_name)
        model_configs.append((model_name, epochs, batch_size, lr, optimizer_type, save_models))
    return model_configs

def clean_results(ablation_results):
    """Clean up results for compatibility"""
    clean_results = {}
    for model_name, result in ablation_results.items():
        clean_result = {k: v for k, v in result.items() 
                       if k not in ['model_name', 'device', 'visualization_dir']}
        clean_results[model_name] = clean_result
    return clean_results 