#!/usr/bin/env python3
"""
Comprehensive ablation study for traffic sign recognition models
Supports parallel training and comprehensive visualization
"""

from pathlib import Path
import time
import json
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Set multiprocessing start method for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import modular components
from ablation.ablation_utils import (
    train_single_model, determine_parallelism, get_default_models, 
    prepare_model_configs, clean_results
)
from ablation.ablation_visualizer import generate_ablation_comparison_with_visualizations


def run_ablation_study_parallel(
    models_to_test=None, epochs=10, batch_size=32, save_models=False,
    max_parallel=4, use_gpu=True, root_path="."
):
    """Run ablation study with parallel training"""
    
    root_path = Path(root_path)
    models_to_test = models_to_test or get_default_models()
    
    # Determine parallelism
    max_parallel = determine_parallelism(max_parallel, use_gpu, len(models_to_test))
    
    print("STARTING PARALLEL ABLATION STUDY")
    print("=" * 60)
    print(f"Models to test: {len(models_to_test)}")
    print(f"Epochs per model: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Max parallel: {max_parallel}")
    print("=" * 60)
    
    # Prepare training configurations
    train_json_path = root_path / "data/annotations/train.json"
    test_json_path = root_path / "data/annotations/test.json"
    shared_data = (root_path, train_json_path, test_json_path)
    model_configs = prepare_model_configs(models_to_test, epochs, batch_size, save_models)
    
    # Execute parallel training
    ablation_results = {}
    failed_models = []
    
    print(f"\nLAUNCHING PARALLEL TRAINING...")
    start_time = time.time()
    
    if use_gpu and torch.cuda.is_available():
        # GPU-based parallel execution
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            future_to_config = {}
            for i, config in enumerate(model_configs):
                gpu_id = i % torch.cuda.device_count()
                future = executor.submit(train_single_model, config, shared_data, gpu_id)
                future_to_config[future] = config
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                model_name = config[0]
                try:
                    result = future.result()
                    if 'error' in result:
                        failed_models.append(model_name)
                        print(f"{model_name} failed: {result['error']}")
                    else:
                        ablation_results[model_name] = result
                except Exception as e:
                    failed_models.append(model_name)
                    print(f"{model_name} failed with exception: {str(e)}")
    else:
        # CPU-based parallel execution
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            future_to_config = {
                executor.submit(train_single_model, config, shared_data): config
                for config in model_configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                model_name = config[0]
                try:
                    result = future.result()
                    if 'error' in result:
                        failed_models.append(model_name)
                        print(f"{model_name} failed: {result['error']}")
                    else:
                        ablation_results[model_name] = result
                except Exception as e:
                    failed_models.append(model_name)
                    print(f"{model_name} failed with exception: {str(e)}")
    
    total_time = time.time() - start_time
    
    # Results summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETED")
    print("=" * 60)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Successful models: {len(ablation_results)}")
    print(f"Failed models: {len(failed_models)}")
    
    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
    
    # Generate comprehensive comparison with enhanced visualizations
    if len(ablation_results) > 1:
        print(f"\nGENERATING COMPREHENSIVE VISUALIZATIONS...")
        clean_results_dict = clean_results(ablation_results)
        generate_ablation_comparison_with_visualizations(clean_results_dict, root_path, ablation_results)
    
    # Save results to JSON
    results_path = root_path / "results/ablation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    return ablation_results


def run_ablation_study_sequential(
    models_to_test=None, epochs=10, batch_size=32, save_models=False,
    root_path="."
):
    """Run ablation study sequentially (original implementation for compatibility)"""
    
    root_path = Path(root_path)
    models_to_test = models_to_test or get_default_models()
    
    print("STARTING SEQUENTIAL ABLATION STUDY")
    print("=" * 60)
    print(f"Models to test: {len(models_to_test)}")
    print(f"Epochs per model: {epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Prepare shared data
    train_json_path = root_path / "data/annotations/train.json"
    test_json_path = root_path / "data/annotations/test.json"
    shared_data = (root_path, train_json_path, test_json_path)
    model_configs = prepare_model_configs(models_to_test, epochs, batch_size, save_models)
    
    # Execute sequential training
    ablation_results = {}
    start_time = time.time()
    
    for config in model_configs:
        model_name = config[0]
        print(f"\nTraining {model_name}...")
        try:
            result = train_single_model(config, shared_data)
            if 'error' in result:
                print(f"{model_name} failed: {result['error']}")
            else:
                ablation_results[model_name] = result
        except Exception as e:
            print(f"{model_name} failed with exception: {str(e)}")
    
    total_time = time.time() - start_time
    
    # Results and visualization
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETED")
    print("=" * 60)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Successful models: {len(ablation_results)}")
    
    # Generate visualizations
    if len(ablation_results) > 1:
        print(f"\nGENERATING COMPREHENSIVE VISUALIZATIONS...")
        clean_results_dict = clean_results(ablation_results)
        generate_ablation_comparison_with_visualizations(clean_results_dict, root_path, ablation_results)
    
    # Save results
    results_path = root_path / "results/ablation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    return ablation_results


def quick_ablation_study(root_path=".", parallel=True):
    """Quick ablation study with reduced models and epochs for testing"""
    quick_models = ['light_cnn', 'simple_cnn', 'wide_cnn']
    print("Running quick ablation study (3 models, 5 epochs each)...")
    
    if parallel:
        return run_ablation_study_parallel(
            models_to_test=quick_models, epochs=5, batch_size=64,
            max_parallel=3, root_path=root_path
        )
    else:
        return run_ablation_study_sequential(
            models_to_test=quick_models, epochs=5, batch_size=64,
            root_path=root_path
        )


def full_ablation_study(root_path=".", parallel=True):
    """Full ablation study with all models"""
    print("Running full ablation study (8 models, 20 epochs each)...")
    
    if parallel:
        return run_ablation_study_parallel(
            epochs=20, batch_size=32, save_models=True,
            max_parallel=4, root_path=root_path
        )
    else:
        return run_ablation_study_sequential(
            epochs=20, batch_size=32, save_models=True,
            root_path=root_path
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--mode', choices=['quick', 'full', 'custom'], default='quick',
                        help='Study mode: quick (3 models, 5 epochs), full (all models, 20 epochs), or custom')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Use parallel training (default: True)')
    parser.add_argument('--sequential', action='store_true', 
                        help='Force sequential training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (for custom mode)')
    parser.add_argument('--models', nargs='+', 
                        help='Specific models to test (for custom mode)')
    
    args = parser.parse_args()
    
    # Determine parallel vs sequential
    use_parallel = args.parallel and not args.sequential
    
    if args.mode == 'quick':
        quick_ablation_study(parallel=use_parallel)
    elif args.mode == 'full':
        full_ablation_study(parallel=use_parallel)
    else:  # custom mode
        if use_parallel:
            run_ablation_study_parallel(
                models_to_test=args.models,
                epochs=args.epochs
            )
        else:
            run_ablation_study_sequential(
                models_to_test=args.models,
                epochs=args.epochs
            ) 