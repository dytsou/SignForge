from .base_visualizer import BaseVisualizer
from .training_plots import TrainingPlots
from .classification_plots import ClassificationPlots
from .comparison_plots import ComparisonPlots

class TrainingVisualizer(BaseVisualizer):
    """Unified interface for all visualization functionality
    
    This class provides access to all visualization capabilities:
    - BaseVisualizer: Core functionality and metric storage
    - TrainingPlots: Training curves, learning rate schedules, loss landscapes
    - ClassificationPlots: Confusion matrices, per-class accuracy, precision/recall
    - ComparisonPlots: Model comparison, ablation studies, Pareto frontiers
    
    Note: Report generation functionality is now available via visualization.sh script
    """
    
    def __init__(self, save_dir="results/visualizations"):
        super().__init__(save_dir)
        
        # Create component instances that share the same save_dir and data
        self._training_plots = TrainingPlots(save_dir)
        self._classification_plots = ClassificationPlots(save_dir)
        self._comparison_plots = ComparisonPlots(save_dir)
        
    def _sync_data_to_components(self):
        """Synchronize data across all component visualizers"""
        for component in [self._training_plots, self._classification_plots, self._comparison_plots]:
            component.epochs = self.epochs
            component.train_losses = self.train_losses
            component.val_accuracies = self.val_accuracies
            component.epoch_times = self.epoch_times
            component.class_names = self.class_names
            component.model_profile = self.model_profile
            
    def log_metrics(self, epoch, train_loss, val_accuracy, epoch_time=None):
        """Log training metrics for an epoch"""
        super().log_metrics(epoch, train_loss, val_accuracy, epoch_time)
        self._sync_data_to_components()
        
    def set_model_profile(self, model_profile):
        """Set model characteristics for ablation study"""
        super().set_model_profile(model_profile)
        self._sync_data_to_components()
        
    # TrainingPlots methods
    def plot_training_curves(self, save_path=None):
        """Plot training loss and validation accuracy curves"""
        self._sync_data_to_components()
        return self._training_plots.plot_training_curves(save_path)
        
    def plot_learning_rate_schedule(self, learning_rates, save_path=None):
        """Plot learning rate schedule over epochs"""
        self._sync_data_to_components()
        return self._training_plots.plot_learning_rate_schedule(learning_rates, save_path)
        
    def plot_loss_landscape(self, train_losses, val_losses=None, save_path=None):
        """Plot training and validation loss landscape"""
        self._sync_data_to_components()
        return self._training_plots.plot_loss_landscape(train_losses, val_losses, save_path)
        
    def plot_gradient_norms(self, gradient_norms, save_path=None):
        """Plot gradient norms over training"""
        self._sync_data_to_components()
        return self._training_plots.plot_gradient_norms(gradient_norms, save_path)
    
    # ClassificationPlots methods
    def plot_confusion_matrix(self, model, test_loader, device, class_names=None, save_path=None):
        """Generate and plot confusion matrix"""
        self._sync_data_to_components()
        return self._classification_plots.plot_confusion_matrix(model, test_loader, device, class_names, save_path)
        
    def plot_class_accuracy(self, model, test_loader, device, class_names=None, save_path=None):
        """Plot per-class accuracy"""
        self._sync_data_to_components()
        return self._classification_plots.plot_class_accuracy(model, test_loader, device, class_names, save_path)
        
    def plot_precision_recall_per_class(self, model, test_loader, device, class_names=None, save_path=None):
        """Plot precision and recall for each class"""
        self._sync_data_to_components()
        return self._classification_plots.plot_precision_recall_per_class(model, test_loader, device, class_names, save_path)
        
    def plot_classification_report_heatmap(self, model, test_loader, device, class_names=None, save_path=None):
        """Plot classification report as heatmap"""
        self._sync_data_to_components()
        return self._classification_plots.plot_classification_report_heatmap(model, test_loader, device, class_names, save_path)
        
    def plot_top_k_accuracy(self, model, test_loader, device, k_values=[1, 3, 5], save_path=None):
        """Plot top-k accuracy for different k values"""
        self._sync_data_to_components()
        return self._classification_plots.plot_top_k_accuracy(model, test_loader, device, k_values, save_path)
    
    # ComparisonPlots methods
    def plot_model_comparison(self, metrics_dict, save_path=None):
        """Compare multiple models' performance"""
        self._sync_data_to_components()
        return self._comparison_plots.plot_model_comparison(metrics_dict, save_path)
        
    def plot_ablation_study(self, ablation_results, save_path=None):
        """Create comprehensive ablation study visualizations"""
        self._sync_data_to_components()
        return self._comparison_plots.plot_ablation_study(ablation_results, save_path)
        
    def plot_pareto_frontier(self, ablation_results, x_metric='parameters_millions', y_metric='accuracy', save_path=None):
        """Plot Pareto frontier for two metrics"""
        self._sync_data_to_components()
        return self._comparison_plots.plot_pareto_frontier(ablation_results, x_metric, y_metric, save_path)
        
    def plot_radar_chart(self, model_metrics, metrics_to_plot=None, save_path=None):
        """Create radar chart for model comparison"""
        self._sync_data_to_components()
        return self._comparison_plots.plot_radar_chart(model_metrics, metrics_to_plot, save_path)
        
    def plot_training_progression_comparison(self, models_data, save_path=None):
        """Plot training progression comparison across models"""
        self._sync_data_to_components()
        return self._comparison_plots.plot_training_progression_comparison(models_data, save_path)
    
    # Legacy method for backward compatibility
    def generate_training_report(self, model, test_loader, device, model_name, class_names=None):
        """Generate a comprehensive training report with multiple visualizations"""
        self._sync_data_to_components()
        
        print(f"Generating comprehensive training report for {model_name}...")
        
        # Generate training plots
        if self.epochs:
            self.plot_training_curves()
            
        # Generate classification plots
        self.plot_confusion_matrix(model, test_loader, device, class_names)
        self.plot_class_accuracy(model, test_loader, device, class_names)
        
        # Generate precision/recall analysis
        self.plot_precision_recall_per_class(model, test_loader, device, class_names)
        
        print(f"Training report for {model_name} completed!")
        print(f"All visualizations saved to: {self.save_dir}") 