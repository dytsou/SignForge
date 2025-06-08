import matplotlib.pyplot as plt
from visualization.base_visualizer import BaseVisualizer

class TrainingPlots(BaseVisualizer):
    """Handles training-related visualizations"""
    
    def plot_training_curves(self, save_path=None):
        """Plot training loss and validation accuracy curves"""
        if not self.epochs:
            print("No training data to plot")
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot training loss
        axes[0].plot(self.epochs, self.train_losses, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(left=0)
        
        # Plot validation accuracy
        axes[1].plot(self.epochs, self.val_accuracies, 'r-', linewidth=2, marker='s', markersize=4)
        axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(left=0)
        axes[1].set_ylim(0, 100)
        
        # Plot training time per epoch (if available)
        if self.epoch_times:
            axes[2].plot(self.epochs[:len(self.epoch_times)], self.epoch_times, 'g-', 
                        linewidth=2, marker='^', markersize=4)
            axes[2].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Time (seconds)')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xlim(left=0)
        else:
            axes[2].text(0.5, 0.5, 'Training time\nnot tracked', ha='center', va='center',
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self._ensure_save_path(save_path, "training_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training curves saved to {save_path}")
        
    def plot_learning_rate_schedule(self, learning_rates, save_path=None):
        """Plot learning rate schedule over epochs"""
        if not learning_rates:
            print("No learning rate data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs[:len(learning_rates)], learning_rates, 'purple', linewidth=2, marker='d', markersize=4)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        save_path = self._ensure_save_path(save_path, "learning_rate_schedule.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Learning rate schedule saved to {save_path}")
        
    def plot_loss_landscape(self, train_losses, val_losses=None, save_path=None):
        """Plot training and validation loss landscape"""
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
        
        if val_losses:
            plt.plot(epochs[:len(val_losses)], val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            plt.legend()
        
        plt.title('Loss Landscape', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        
        save_path = self._ensure_save_path(save_path, "loss_landscape.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Loss landscape saved to {save_path}")
        
    def plot_gradient_norms(self, gradient_norms, save_path=None):
        """Plot gradient norms over training"""
        if not gradient_norms:
            print("No gradient norm data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(gradient_norms)), gradient_norms, 'orange', linewidth=2, alpha=0.7)
        plt.title('Gradient Norms During Training', fontsize=14, fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        save_path = self._ensure_save_path(save_path, "gradient_norms.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Gradient norms plot saved to {save_path}") 