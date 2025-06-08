"""
Visualization functions for ablation study
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import seaborn as sns
import torch
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import utils

def create_model_summary_plot(model_name, model_profile, model_size, accuracy, training_time, save_path):
    """Create a 4-panel summary plot for a single model"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Model Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    # Model Characteristics (Bar chart)
    metrics = ['Parameters (M)', 'Size (MB)', 'Inference (ms)']
    values = [model_profile['parameters_millions'], model_profile['model_size_mb'], 
              model_profile['inference_time_ms']]
    bars = ax1.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Model Characteristics')
    ax1.set_ylabel('Value')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    # Performance Radar Chart
    categories = ['Accuracy', 'Speed', 'Efficiency', 'Size']
    values_norm = [
        accuracy / 100,  # Normalize to 0-1
        max(0, 1 - model_profile['inference_time_ms'] / 100),  # Inverse time
        max(0, 1 - model_profile['parameters_millions'] / 50),  # Inverse parameters
        max(0, 1 - model_size / 100)  # Inverse size
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_norm += values_norm[:1]  # Complete the circle
    angles += angles[:1]
    
    ax2.plot(angles, values_norm, 'o-', linewidth=2, color='blue')
    ax2.fill(angles, values_norm, alpha=0.25, color='blue')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('Performance Profile')
    ax2.grid(True)
    
    # Efficiency Plot
    efficiency_score = accuracy / (model_profile['parameters_millions'] + 1)
    ax3.bar(['Efficiency Score'], [efficiency_score], color='orange')
    ax3.set_title('Accuracy/Parameters Ratio')
    ax3.set_ylabel('Score')
    ax3.text(0, efficiency_score, f'{efficiency_score:.2f}', 
             ha='center', va='bottom', fontweight='bold')
    
    # Training Summary
    summary_text = f"""
    Model: {model_name}
    Accuracy: {accuracy:.2f}%
    Parameters: {model_profile['parameters_millions']:.2f}M
    Model Size: {model_size:.2f} MB
    Training Time: {training_time:.1f}s
    Inference: {model_profile['inference_time_ms']:.2f}ms
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_training_summary_plot(visualizer, model_name, save_path):
    """Create training progress summary plot"""
    if not visualizer.train_losses:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Summary: {model_name}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(visualizer.train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, visualizer.train_losses, 'b-', label='Train Loss', linewidth=2)
    if visualizer.val_losses:
        ax1.plot(epochs, visualizer.val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    if visualizer.val_accuracies:
        ax2.plot(epochs, visualizer.val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if hasattr(visualizer, 'learning_rates') and visualizer.learning_rates:
        ax3.plot(epochs, visualizer.learning_rates, 'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Learning Rate Schedule')
    
    # Training metrics summary
    best_acc = max(visualizer.val_accuracies) if visualizer.val_accuracies else 0
    final_loss = visualizer.train_losses[-1] if visualizer.train_losses else 0
    
    metrics_text = f"""
    Best Validation Accuracy: {best_acc:.2f}%
    Final Training Loss: {final_loss:.4f}
    Total Epochs: {len(visualizer.train_losses)}
    Converged: {'Yes' if len(visualizer.train_losses) > 5 else 'No'}
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Training Metrics')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_individual_visualizations(model, test_loader, device, model_name, visualizer, model_profile, save_dir):
    """Save comprehensive visualizations for individual model"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Model summary plot
    accuracy = utils.evaluate(model, test_loader, device, silent=True)
    training_time = getattr(visualizer, 'total_training_time', 0)
    
    create_model_summary_plot(
        model_name, model_profile, model_profile['model_size_mb'], 
        accuracy, training_time, save_dir / 'model_summary.png'
    )
    
    # Training summary plot
    create_training_summary_plot(visualizer, model_name, save_dir / 'training_summary.png')
    
    print(f"   Individual visualizations saved to {save_dir}")

def create_bubble_chart_analysis(results, save_path):
    """Create bubble chart for comprehensive model comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_name, result in results.items():
        x = result['parameters_millions']
        y = result['accuracy']
        size = result['model_size_mb'] * 10  # Scale for visibility
        color = result['total_training_time']
        
        scatter = ax.scatter(x, y, s=size, c=color, alpha=0.6, 
                           cmap='viridis', edgecolors='black', linewidth=1)
        ax.annotate(model_name, (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison: Accuracy vs Parameters\n(Bubble size = Model Size, Color = Training Time)')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Training Time (seconds)')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_ranking_charts(results, save_path):
    """Create 4-panel ranking comparison charts"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Rankings Comparison', fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    
    # Accuracy ranking
    acc_data = [(name, results[name]['accuracy']) for name in models]
    acc_data.sort(key=lambda x: x[1], reverse=True)
    names, values = zip(*acc_data)
    bars1 = ax1.barh(names, values, color='skyblue')
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Accuracy Ranking')
    for i, v in enumerate(values):
        ax1.text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    # Parameter efficiency ranking (accuracy per million parameters)
    eff_data = [(name, results[name]['accuracy'] / results[name]['parameters_millions']) 
                for name in models]
    eff_data.sort(key=lambda x: x[1], reverse=True)
    names, values = zip(*eff_data)
    bars2 = ax2.barh(names, values, color='lightcoral')
    ax2.set_xlabel('Accuracy per Million Parameters')
    ax2.set_title('Parameter Efficiency Ranking')
    for i, v in enumerate(values):
        ax2.text(v + 0.1, i, f'{v:.1f}', va='center')
    
    # Training speed ranking (inverse of training time)
    speed_data = [(name, 1/results[name]['total_training_time']) for name in models]
    speed_data.sort(key=lambda x: x[1], reverse=True)
    names, values = zip(*speed_data)
    bars3 = ax3.barh(names, values, color='lightgreen')
    ax3.set_xlabel('Training Speed (1/seconds)')
    ax3.set_title('Training Speed Ranking')
    for i, v in enumerate(values):
        ax3.text(v + 0.0001, i, f'{v:.4f}', va='center')
    
    # Model size ranking (smaller is better)
    size_data = [(name, results[name]['model_size_mb']) for name in models]
    size_data.sort(key=lambda x: x[1])  # Ascending for size
    names, values = zip(*size_data)
    bars4 = ax4.barh(names, values, color='orange')
    ax4.set_xlabel('Model Size (MB)')
    ax4.set_title('Model Size Ranking (Lower is Better)')
    for i, v in enumerate(values):
        ax4.text(v + 0.5, i, f'{v:.1f}MB', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_html_gallery(results, base_path):
    """Generate HTML gallery for easy viewing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ablation Study Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; }
            .header { text-align: center; margin-bottom: 30px; }
            .model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .model-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fafafa; }
            .model-card h3 { margin-top: 0; color: #333; }
            .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            .metric { background: white; padding: 8px; border-radius: 4px; text-align: center; }
            .images { margin-top: 15px; }
            .images img { max-width: 100%; height: auto; margin: 5px 0; border-radius: 4px; }
            .comparison { margin-top: 30px; text-align: center; }
            .comparison img { max-width: 100%; height: auto; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Traffic Sign Recognition - Ablation Study Results</h1>
                <p>Comprehensive analysis of model architectures and their performance</p>
            </div>
    """
    
    # Add individual model cards
    html_content += '<div class="model-grid">'
    
    for model_name, result in results.items():
        viz_dir = Path(base_path) / "results" / "ablation" / model_name
        html_content += f"""
            <div class="model-card">
                <h3>{model_name}</h3>
                <div class="metrics">
                    <div class="metric"><strong>Accuracy:</strong> {result.get('accuracy', 0):.2f}%</div>
                    <div class="metric"><strong>Parameters:</strong> {result.get('parameters_millions', 0):.2f}M</div>
                    <div class="metric"><strong>Size:</strong> {result.get('model_size_mb', 0):.2f} MB</div>
                    <div class="metric"><strong>Training Time:</strong> {result.get('total_training_time', 0):.1f}s</div>
                </div>
                <div class="images">
        """
        
        # Add images if they exist
        for img_name in ['model_summary.png', 'training_summary.png']:
            img_path = viz_dir / img_name
            if img_path.exists():
                rel_path = img_path.relative_to(base_path)
                html_content += f'<img src="{rel_path}" alt="{img_name}">'
        
        html_content += '</div></div>'
    
    html_content += '</div>'
    
    # Add comparison section
    comparison_dir = Path(base_path) / "results" / "comparison"
    html_content += '<div class="comparison">'
    html_content += '<h2>Model Comparisons</h2>'
    
    for img_name in ['bubble_analysis.png', 'ranking_comparison.png', 'ablation_study.png']:
        img_path = comparison_dir / img_name
        if img_path.exists():
            rel_path = img_path.relative_to(base_path)
            html_content += f'<img src="{rel_path}" alt="{img_name}">'
    
    html_content += '</div></div></body></html>'
    
    # Save HTML file
    html_path = Path(base_path) / "results" / "ablation_study_gallery.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"   HTML gallery generated: {html_path}")
    return html_path

def generate_ablation_comparison_with_visualizations(results, root_path, full_results):
    """Generate comprehensive comparison with enhanced visualizations"""
    comparison_dir = Path(root_path) / "results" / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate enhanced visualization components
    create_bubble_chart_analysis(results, comparison_dir / "bubble_analysis.png")
    create_ranking_charts(results, comparison_dir / "ranking_comparison.png")
    
    # Generate the main ablation study plot (from existing code)
    from analysis.compare_models import generate_ablation_comparison
    generate_ablation_comparison(full_results, root_path)
    
    # Generate HTML gallery
    generate_html_gallery(full_results, root_path)
    
    print(f"Enhanced visualizations saved to {comparison_dir}") 