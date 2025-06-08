import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from pathlib import Path
import json
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.dataset import GTSRBDataset
from core import utils
from visualization.visualizer import TrainingVisualizer
from core.models import SimpleCNN, ModelProfiler, get_model

def load_model_metrics(metrics_path):
    """Load saved training metrics from JSON file"""
    if Path(metrics_path).exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def load_ablation_results(root_path):
    """Load ablation study results if available"""
    ablation_path = root_path / "results/ablation_comparison/detailed_results.json"
    if ablation_path.exists():
        with open(ablation_path, 'r') as f:
            return json.load(f)
    return None

def evaluate_saved_model(model_class, checkpoint_path, test_loader, device, model_name):
    """Evaluate a saved model checkpoint"""
    model = model_class()
    if checkpoint_path.exists():
        model = utils.load_checkpoint(model, checkpoint_path, device)
        model.to(device)
        acc = utils.evaluate(model, test_loader, device, f"[{model_name}]")
        
        # Profile the model
        model_profile = ModelProfiler.get_model_profile(model, device=device)
        
        return acc, model, model_profile
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None, None

def generate_ablation_comparison(ablation_results, root_path):
    """Generate ablation comparison plots (placeholder for now)"""
    # This function would generate the main ablation comparison plot
    # For now, we'll use the visualizer's ablation study plot
    visualizer = TrainingVisualizer(save_dir=root_path/"results/comparison")
    
    # Convert ablation results to the format expected by the visualizer
    formatted_results = {}
    for model_name, result in ablation_results.items():
        if 'error' not in result:
            formatted_results[model_name] = {
                'accuracy': result.get('accuracy', 0),
                'parameters_millions': result.get('parameters_millions', 0),
                'model_size_mb': result.get('model_size_mb', 0),
                'inference_time_ms': result.get('inference_time_ms', 0),
                'avg_epoch_time': result.get('avg_epoch_time', 0)
            }
    
    if formatted_results:
        visualizer.plot_ablation_study(formatted_results)

def main():
    root = Path(__file__).resolve().parents[2]
    
    # Setup test data
    test_ds = GTSRBDataset(root/"data/annotations/test.json", root, False)
    test_loader = DataLoader(test_ds, 128, shuffle=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize comparison visualizer
    visualizer = TrainingVisualizer(save_dir=root/"results/visualizations/comparison")
    
    print("="*60)
    print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Check for ablation study results first
    ablation_data = load_ablation_results(root)
    
    if ablation_data:
        print("\nABLATION STUDY RESULTS FOUND!")
        print("Using comprehensive ablation study data...")
        
        ablation_results = ablation_data['ablation_results']
        summary = ablation_data['summary']
        
        # Display comprehensive comparison
        print(f"\nCOMPREHENSIVE ABLATION STUDY COMPARISON")
        print("-" * 100)
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(ablation_results).T
        df = df.sort_values('accuracy', ascending=False)
        
        print(f"{'Model':<20} {'Acc%':<8} {'Params(M)':<12} {'Size(MB)':<10} {'Inf(ms)':<10} {'Train(s/ep)':<12}")
        print("-" * 100)
        
        for model_name, row in df.iterrows():
            print(f"{model_name:<20} {row['accuracy']:<8.2f} {row['parameters_millions']:<12.2f} "
                  f"{row['model_size_mb']:<10.2f} {row['inference_time_ms']:<10.2f} "
                  f"{row['avg_epoch_time']:<12.1f}")
        
        # Display key insights
        print(f"\nKEY FINDINGS:")
        print(f"   Best Accuracy: {summary['best_accuracy']['model']} ({summary['best_accuracy']['value']:.2f}%)")
        print(f"   Fastest Training: {summary['fastest_training']['model']} ({summary['fastest_training']['value']:.1f}s/epoch)")
        print(f"   Smallest Model: {summary['smallest_model']['model']} ({summary['smallest_model']['value']:.2f} MB)")
        print(f"   Fastest Inference: {summary['fastest_inference']['model']} ({summary['fastest_inference']['value']:.2f} ms)")
        print(f"   Most Efficient: {summary['most_efficient']['model']} (score: {summary['most_efficient']['value']:.2f})")
        
        # Generate enhanced visualizations
        visualizer.plot_ablation_study(ablation_results)
        
        # Generate recommendation
        print(f"\nRECOMMENDATIONS:")
        
        # Production deployment recommendation
        efficiency_scores = ablation_data.get('efficiency_scores', {})
        if efficiency_scores:
            top_efficient = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   For Production: {top_efficient[0][0]} (best efficiency)")
            
        # Research/experimentation recommendation
        best_acc_model = summary['best_accuracy']['model']
        print(f"   For Research: {best_acc_model} (highest accuracy)")
        
        # Mobile/edge deployment recommendation
        mobile_candidates = [(k, v) for k, v in ablation_results.items() 
                           if v['model_size_mb'] < 10 and v['inference_time_ms'] < 5]
        if mobile_candidates:
            best_mobile = max(mobile_candidates, key=lambda x: x[1]['accuracy'])
            print(f"   For Mobile: {best_mobile[0]} (lightweight + accurate)")
        
    else:
        print("\nSTANDARD MODEL COMPARISON")
        print("No ablation study results found. Comparing individual trained models...")
        
        # Model comparison dictionary
        model_metrics = {}
        
        # Evaluate Baseline CNN
        print("\n1. Evaluating Baseline CNN...")
        baseline_checkpoints = list((root/"models").glob("baseline_ep*.pth"))
        if baseline_checkpoints:
            latest_baseline = max(baseline_checkpoints, key=lambda x: int(x.stem.split('ep')[1]))
            baseline_acc, baseline_model, baseline_profile = evaluate_saved_model(
                lambda: SimpleCNN(), 
                latest_baseline, 
                test_loader, 
                device, 
                "Baseline CNN"
            )
            if baseline_acc is not None:
                model_metrics["Baseline CNN"] = {
                    "accuracy": baseline_acc,
                    **baseline_profile
                }
                
                # Load training history if available
                baseline_metrics_path = root/"results/visualizations/baseline/baseline_cnn_metrics.json"
                baseline_history = load_model_metrics(baseline_metrics_path)
                if baseline_history:
                    model_metrics["Baseline CNN"]["history"] = baseline_history["training_history"]
        
        # Evaluate MobileNet
        print("\n2. Evaluating MobileNetV2...")
        mobilenet_checkpoints = list((root/"models").glob("mobilenet_ep*.pth"))
        if mobilenet_checkpoints:
            latest_mobilenet = max(mobilenet_checkpoints, key=lambda x: int(x.stem.split('ep')[1]))
            
            # Create MobileNet model architecture
            def create_mobilenet():
                model = mobilenet_v2(weights=None)  # Don't load pretrained weights for evaluation
                model.classifier[1] = nn.Linear(model.last_channel, 43)
                return model
                
            mobilenet_acc, mobilenet_model, mobilenet_profile = evaluate_saved_model(
                create_mobilenet,
                latest_mobilenet,
                test_loader,
                device,
                "MobileNetV2"
            )
            if mobilenet_acc is not None:
                model_metrics["MobileNetV2"] = {
                    "accuracy": mobilenet_acc,
                    **mobilenet_profile
                }
                
                # Load training history if available
                mobilenet_metrics_path = root/"results/visualizations/mobilenet/mobilenet_v2_metrics.json"
                mobilenet_history = load_model_metrics(mobilenet_metrics_path)
                if mobilenet_history:
                    model_metrics["MobileNetV2"]["history"] = mobilenet_history["training_history"]
        
        # Check for other ablation models
        ablation_models = list((root/"models").glob("ablation_*.pth"))
        for model_path in ablation_models:
            model_name = model_path.stem.replace('ablation_', '')
            print(f"\n3. Evaluating {model_name}...")
            
            try:
                model = get_model(model_name)
                model = utils.load_checkpoint(model, model_path, device)
                model.to(device)
                
                acc = utils.evaluate(model, test_loader, device, f"[{model_name}]")
                profile = ModelProfiler.get_model_profile(model, device=device)
                
                model_metrics[model_name] = {
                    "accuracy": acc,
                    **profile
                }
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        # Generate comparison visualizations
        if len(model_metrics) >= 2:
            print("\n3. Generating Model Comparison Visualizations...")
            
            # Plot accuracy comparison
            visualizer.plot_model_comparison(model_metrics)
            
            # If we have enough data, create mini ablation study
            if len(model_metrics) >= 3:
                # Convert to ablation format
                ablation_format = {}
                for model_name, metrics in model_metrics.items():
                    ablation_format[model_name] = {
                        'accuracy': metrics['accuracy'],
                        'parameters_millions': metrics.get('parameters_millions', 0),
                        'model_size_mb': metrics.get('model_size_mb', 0),
                        'inference_time_ms': metrics.get('inference_time_ms', 0),
                        'avg_epoch_time': 0  # Not available for loaded models
                    }
                
                visualizer.plot_ablation_study(ablation_format)
            
            # Generate detailed comparison report
            print("\n4. Performance Summary:")
            print("-" * 60)
            print(f"{'Model':<20} {'Accuracy':<12} {'Parameters':<12} {'Size(MB)':<10}")
            print("-" * 60)
            
            for model_name, metrics in sorted(model_metrics.items(), 
                                            key=lambda x: x[1]['accuracy'], reverse=True):
                print(f"{model_name:<20} {metrics['accuracy']:<12.2f} "
                      f"{metrics.get('parameters_millions', 0):<12.2f} "
                      f"{metrics.get('model_size_mb', 0):<10.2f}")
            
            best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
            print(f"\nBest performing model: {best_model[0]} ({best_model[1]['accuracy']:.2f}%)")
            
        else:
            print("\nNot enough trained models found for comparison.")
            print("To run a comprehensive ablation study:")
            print("  python src/ablation/ablation_study.py")
            print("Or train individual models:")
            print("  python src/training/train_baseline.py")
            print("  python src/training/train_mobilenet.py")

if __name__ == "__main__":
    main() 