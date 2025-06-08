import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from visualization.base_visualizer import BaseVisualizer

class ClassificationPlots(BaseVisualizer):
    """Handles classification-specific visualizations"""
    
    def plot_confusion_matrix(self, model, test_loader, device, class_names=None, save_path=None):
        """Generate and plot confusion matrix"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names if class_names else range(len(cm)),
                   yticklabels=class_names if class_names else range(len(cm)))
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        save_path = self._ensure_save_path(save_path, "confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to {save_path}")
        
        return cm, all_preds, all_labels
    
    def plot_class_accuracy(self, model, test_loader, device, class_names=None, save_path=None):
        """Plot per-class accuracy"""
        cm, preds, labels = self.plot_confusion_matrix(model, test_loader, device, class_names, 
                                                      save_path=self.save_dir / "temp_cm.png")
        
        # Calculate per-class accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(15, 8))
        x_pos = np.arange(len(class_accuracies))
        bars = plt.bar(x_pos, class_accuracies * 100, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.xticks(x_pos, class_names if class_names else range(len(class_accuracies)), 
                  rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        save_path = self._ensure_save_path(save_path, "class_accuracy.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Class accuracy plot saved to {save_path}")
        
        # Clean up temp file
        temp_path = self.save_dir / "temp_cm.png"
        if temp_path.exists():
            temp_path.unlink()
            
        return class_accuracies
    
    def plot_precision_recall_per_class(self, model, test_loader, device, class_names=None, save_path=None):
        """Plot precision and recall for each class"""
        cm, preds, labels = self.plot_confusion_matrix(model, test_loader, device, class_names,
                                                      save_path=self.save_dir / "temp_cm.png")
        
        # Calculate precision and recall for each class
        precision = cm.diagonal() / cm.sum(axis=0)
        recall = cm.diagonal() / cm.sum(axis=1)
        
        x = np.arange(len(class_names) if class_names else len(precision))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(15, 8))
        bars1 = ax.bar(x - width/2, precision * 100, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x + width/2, recall * 100, width, label='Recall', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score (%)')
        ax.set_title('Precision and Recall per Class')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names if class_names else range(len(precision)), rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        save_path = self._ensure_save_path(save_path, "precision_recall_per_class.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Precision-Recall plot saved to {save_path}")
        
        # Clean up temp file
        temp_path = self.save_dir / "temp_cm.png"
        if temp_path.exists():
            temp_path.unlink()
            
        return precision, recall
    
    def plot_classification_report_heatmap(self, model, test_loader, device, class_names=None, save_path=None):
        """Plot classification report as heatmap"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # Generate classification report
        target_names = class_names if class_names else [f'Class {i}' for i in range(max(all_labels) + 1)]
        report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
        
        # Extract metrics for heatmap
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        labels = []
        
        for class_name in target_names:
            if class_name in report:
                data.append([report[class_name][metric] for metric in metrics])
                labels.append(class_name)
        
        data = np.array(data)
        
        plt.figure(figsize=(8, len(labels) * 0.5 + 2))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=metrics, yticklabels=labels,
                   cbar_kws={'label': 'Score'})
        plt.title('Classification Report Heatmap')
        plt.tight_layout()
        
        save_path = self._ensure_save_path(save_path, "classification_report_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Classification report heatmap saved to {save_path}")
        
        return report
    
    def plot_top_k_accuracy(self, model, test_loader, device, k_values=[1, 3, 5], save_path=None):
        """Plot top-k accuracy for different k values"""
        model.eval()
        correct_counts = {k: 0 for k in k_values}
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                
                for k in k_values:
                    _, top_k_preds = outputs.topk(k, dim=1)
                    correct_counts[k] += (top_k_preds == y.unsqueeze(1)).any(dim=1).sum().item()
                
                total += y.size(0)
        
        # Calculate accuracies
        accuracies = [correct_counts[k] / total * 100 for k in k_values]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([f'Top-{k}' for k in k_values], accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Top-K Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        save_path = self._ensure_save_path(save_path, "top_k_accuracy.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Top-K accuracy plot saved to {save_path}")
        
        return dict(zip(k_values, accuracies)) 