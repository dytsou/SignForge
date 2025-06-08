import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visualization.base_visualizer import BaseVisualizer

class ComparisonPlots(BaseVisualizer):
    """Handles model comparison and ablation study visualizations"""
    
    def plot_model_comparison(self, metrics_dict, save_path=None):
        """Compare multiple models' performance"""
        models = list(metrics_dict.keys())
        accuracies = [metrics_dict[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['lightcoral', 'lightblue', 'lightgreen', 
                                                 'lightyellow', 'lightpink', 'lightgray'])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        save_path = self._ensure_save_path(save_path, "model_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Model comparison saved to {save_path}")
    
    def plot_ablation_study(self, ablation_results, save_path=None):
        """Create comprehensive ablation study visualizations"""
        if not ablation_results:
            print("No ablation results to plot")
            return
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(ablation_results).T
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Accuracy vs Model Size
        axes[0, 0].scatter(df['model_size_mb'], df['accuracy'], s=100, alpha=0.7)
        for i, model in enumerate(df.index):
            axes[0, 0].annotate(model, (df.iloc[i]['model_size_mb'], df.iloc[i]['accuracy']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 0].set_xlabel('Model Size (MB)')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Accuracy vs Model Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy vs Parameters
        axes[0, 1].scatter(df['parameters_millions'], df['accuracy'], s=100, alpha=0.7, color='orange')
        for i, model in enumerate(df.index):
            axes[0, 1].annotate(model, (df.iloc[i]['parameters_millions'], df.iloc[i]['accuracy']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('Parameters (Millions)')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy vs Number of Parameters')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Accuracy vs Training Time
        if 'avg_epoch_time' in df.columns:
            axes[0, 2].scatter(df['avg_epoch_time'], df['accuracy'], s=100, alpha=0.7, color='green')
            for i, model in enumerate(df.index):
                axes[0, 2].annotate(model, (df.iloc[i]['avg_epoch_time'], df.iloc[i]['accuracy']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[0, 2].set_xlabel('Avg Training Time per Epoch (s)')
            axes[0, 2].set_ylabel('Accuracy (%)')
            axes[0, 2].set_title('Accuracy vs Training Speed')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Inference Time vs Accuracy
        if 'inference_time_ms' in df.columns:
            axes[1, 0].scatter(df['inference_time_ms'], df['accuracy'], s=100, alpha=0.7, color='red')
            for i, model in enumerate(df.index):
                axes[1, 0].annotate(model, (df.iloc[i]['inference_time_ms'], df.iloc[i]['accuracy']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 0].set_xlabel('Inference Time (ms)')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title('Accuracy vs Inference Speed')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Parameters vs Model Size
        axes[1, 1].scatter(df['parameters_millions'], df['model_size_mb'], s=100, alpha=0.7, color='purple')
        for i, model in enumerate(df.index):
            axes[1, 1].annotate(model, (df.iloc[i]['parameters_millions'], df.iloc[i]['model_size_mb']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Parameters (Millions)')
        axes[1, 1].set_ylabel('Model Size (MB)')
        axes[1, 1].set_title('Parameters vs Model Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Efficiency Score
        if 'avg_epoch_time' in df.columns:
            efficiency = df['accuracy'] / (df['model_size_mb'] * df['avg_epoch_time'])
            efficiency_df = pd.DataFrame({'model': df.index, 'efficiency': efficiency})
            efficiency_df = efficiency_df.sort_values('efficiency', ascending=True)
            
            bars = axes[1, 2].barh(efficiency_df['model'], efficiency_df['efficiency'])
            axes[1, 2].set_xlabel('Efficiency Score')
            axes[1, 2].set_title('Model Efficiency Ranking')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1, 2].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        save_path = self._ensure_save_path(save_path, "ablation_study.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Ablation study visualization saved to {save_path}")
    
    def plot_pareto_frontier(self, ablation_results, x_metric='parameters_millions', y_metric='accuracy', save_path=None):
        """Plot Pareto frontier for two metrics"""
        if not ablation_results:
            print("No ablation results to plot")
            return
            
        df = pd.DataFrame(ablation_results).T
        
        plt.figure(figsize=(12, 8))
        
        # Plot all models
        plt.scatter(df[x_metric], df[y_metric], s=100, alpha=0.7, color='lightblue', edgecolors='blue')
        
        # Annotate points
        for i, model in enumerate(df.index):
            plt.annotate(model, (df.iloc[i][x_metric], df.iloc[i][y_metric]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Find and highlight Pareto frontier
        # For maximizing y_metric and minimizing x_metric
        pareto_points = []
        sorted_points = df.sort_values(x_metric)
        
        max_y = -np.inf
        for idx, row in sorted_points.iterrows():
            if row[y_metric] > max_y:
                max_y = row[y_metric]
                pareto_points.append((row[x_metric], row[y_metric], idx))
        
        if pareto_points:
            pareto_x, pareto_y, pareto_names = zip(*pareto_points)
            plt.plot(pareto_x, pareto_y, 'r-', linewidth=2, alpha=0.7, label='Pareto Frontier')
            plt.scatter(pareto_x, pareto_y, s=150, color='red', alpha=0.8, edgecolors='darkred', 
                       label='Pareto Optimal', zorder=5)
        
        plt.xlabel(x_metric.replace('_', ' ').title())
        plt.ylabel(y_metric.replace('_', ' ').title())
        plt.title(f'Pareto Frontier: {y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        save_path = self._ensure_save_path(save_path, f"pareto_frontier_{x_metric}_{y_metric}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Pareto frontier plot saved to {save_path}")
        
        return pareto_points
    
    def plot_radar_chart(self, model_metrics, metrics_to_plot=None, save_path=None):
        """Create radar chart for model comparison"""
        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'speed', 'efficiency', 'size']
        
        # Normalize metrics for radar chart (0-1 scale)
        normalized_data = {}
        
        for model_name, metrics in model_metrics.items():
            normalized_data[model_name] = {}
            
            # Normalize accuracy (0-100 -> 0-1)
            normalized_data[model_name]['accuracy'] = metrics.get('accuracy', 0) / 100
            
            # Normalize speed (inverse of time, higher is better)
            if 'avg_epoch_time' in metrics:
                max_time = max([m.get('avg_epoch_time', 1) for m in model_metrics.values()])
                normalized_data[model_name]['speed'] = 1 - (metrics.get('avg_epoch_time', 1) / max_time)
            else:
                normalized_data[model_name]['speed'] = 0.5
            
            # Normalize efficiency (accuracy per parameter)
            if 'parameters_millions' in metrics:
                efficiency = metrics.get('accuracy', 0) / max(metrics.get('parameters_millions', 1), 0.1)
                max_efficiency = max([m.get('accuracy', 0) / max(m.get('parameters_millions', 1), 0.1) 
                                    for m in model_metrics.values()])
                normalized_data[model_name]['efficiency'] = efficiency / max_efficiency
            else:
                normalized_data[model_name]['efficiency'] = 0.5
            
            # Normalize size (inverse, smaller is better)
            if 'model_size_mb' in metrics:
                max_size = max([m.get('model_size_mb', 1) for m in model_metrics.values()])
                normalized_data[model_name]['size'] = 1 - (metrics.get('model_size_mb', 1) / max_size)
            else:
                normalized_data[model_name]['size'] = 0.5
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (model_name, data) in enumerate(normalized_data.items()):
            values = [data[metric] for metric in metrics_to_plot]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                   color=colors[i % len(colors)], alpha=0.7)
            ax.fill(angles, values, alpha=0.2, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_to_plot])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        
        save_path = self._ensure_save_path(save_path, "radar_chart.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Radar chart saved to {save_path}")
    
    def plot_training_progression_comparison(self, models_data, save_path=None):
        """Compare training progression across multiple models"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots for loss and accuracy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (model_name, data) in enumerate(models_data.items()):
            color = colors[i % len(colors)]
            
            # Plot training loss
            if 'train_losses' in data and data['train_losses']:
                epochs = range(1, len(data['train_losses']) + 1)
                ax1.plot(epochs, data['train_losses'], '-', linewidth=2, 
                        label=f'{model_name}', color=color, alpha=0.8)
            
            # Plot validation accuracy
            if 'val_accuracies' in data and data['val_accuracies']:
                epochs = range(1, len(data['val_accuracies']) + 1)
                ax2.plot(epochs, data['val_accuracies'], '-', linewidth=2, 
                        label=f'{model_name}', color=color, alpha=0.8)
        
        # Configure loss subplot
        ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Configure accuracy subplot
        ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        save_path = self._ensure_save_path(save_path, "training_progression_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training progression comparison saved to {save_path}") 