#!/usr/bin/env python3
"""
Demo script showing how to use the visualization features
Run this after training models to see comprehensive visualizations
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.dataset import GTSRBDataset
from core import utils
from .visualizer import TrainingVisualizer
from core.models import SimpleCNN

def demo_visualization():
    """Demonstrate the visualization capabilities"""
    root = Path(__file__).resolve().parents[2]
    
    print("Traffic Sign Recognition - Visualization Demo")
    print("=" * 50)
    
    # Check if trained models exist
    models_dir = root / "models"
    baseline_models = list(models_dir.glob("baseline_ep*.pth"))
    mobilenet_models = list(models_dir.glob("mobilenet_ep*.pth"))
    
    if not baseline_models and not mobilenet_models:
        print("No trained models found!")
        print("\nTo use this demo, first train some models:")
        print("  python src/training/train_baseline.py")
        print("  python src/training/train_mobilenet.py")
        return
    
    # Setup test data
    test_ds = GTSRBDataset(root/"data/annotations/test.json", root, False)
    test_loader = DataLoader(test_ds, 64, shuffle=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    class_names = utils.get_class_names()
    
    print(f"Device: {device}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Classes: {len(class_names)}")
    
    # Demo with baseline model if available
    if baseline_models:
        print("\nAnalyzing Baseline CNN Performance...")
        latest_baseline = max(baseline_models, key=lambda x: int(x.stem.split('ep')[1]))
        
        # Load model
        model = SimpleCNN().to(device)
        model = utils.load_checkpoint(model, latest_baseline, device)
        
        # Initialize visualizer
        visualizer = TrainingVisualizer(save_dir=root/"results/demo_visualization")
        
        # Generate comprehensive analysis
        print("   Generating training curves...")
        print("   Creating confusion matrix...")
        print("   Analyzing per-class accuracy...")
        
        # If we have saved metrics, load them
        metrics_path = root/"results/visualizations/baseline/baseline_cnn_metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                saved_metrics = json.load(f)
            
            # Populate visualizer with saved data
            history = saved_metrics.get("training_history", {})
            if history:
                visualizer.epochs = history.get("epochs", [])
                visualizer.train_losses = history.get("train_losses", [])
                visualizer.val_accuracies = history.get("val_accuracies", [])
        
        # Generate visualizations
        if visualizer.epochs:
            visualizer.plot_training_curves()
        
        visualizer.plot_confusion_matrix(model, test_loader, device, class_names)
        visualizer.plot_class_accuracy(model, test_loader, device, class_names)
        
        print("Baseline analysis complete!")
    
    # Demo model comparison if multiple models exist
    if baseline_models and mobilenet_models:
        print("\nRunning Model Comparison...")
        from analysis.compare_models import main as compare_main
        compare_main()
        print("Model comparison complete!")
    
    print("\nVisualization Demo Complete!")
    print(f"Check results in: {root}/results/")
    print("\nVisualization Features Available:")
    print("  Training loss curves")
    print("  Validation accuracy curves") 
    print("  Confusion matrices")
    print("  Per-class accuracy analysis")
    print("  Model comparison charts")
    print("  Detailed performance reports")

def show_usage():
    """Show how to integrate visualization in training scripts"""
    print("\nHow to Use Visualization in Your Training:")
    print("-" * 45)
    
    code_example = '''
# 1. Import the visualizer
from visualization.visualizer import TrainingVisualizer

# 2. Initialize visualizer
visualizer = TrainingVisualizer(save_dir="results/my_model")

# 3. In your training loop
for epoch in range(epochs):
    # Train and get metrics
    avg_loss = train_one_epoch(...)
    accuracy = evaluate(...)
    
    # Log for visualization
    visualizer.log_metrics(epoch, avg_loss, accuracy)
    
    # Plot curves every few epochs (optional)
    if epoch % 5 == 0:
        visualizer.plot_training_curves()

# 4. Generate final report
class_names = utils.get_class_names()
visualizer.generate_training_report(
    model, test_loader, device, "my_model", class_names
)
'''
    
    print(code_example)

if __name__ == "__main__":
    demo_visualization()
    show_usage() 