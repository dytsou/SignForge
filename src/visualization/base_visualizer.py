import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

class BaseVisualizer:
    """Base class for all visualization functionality"""
    
    def __init__(self, save_dir="results/visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric storage
        self.train_losses = []
        self.val_accuracies = []
        self.epochs = []
        self.epoch_times = []  # Track training time per epoch
        self.class_names = None
        
        # Model characteristics
        self.model_profile = {}
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def log_metrics(self, epoch, train_loss, val_accuracy, epoch_time=None):
        """Log training metrics for an epoch"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_accuracies.append(val_accuracy)
        if epoch_time:
            self.epoch_times.append(epoch_time)
        
    def set_model_profile(self, model_profile):
        """Set model characteristics for ablation study"""
        self.model_profile = model_profile
        
    def _ensure_save_path(self, save_path, default_filename):
        """Ensure save path is valid"""
        if save_path is None:
            save_path = self.save_dir / default_filename
        return save_path
        
    def _setup_plot_style(self):
        """Setup consistent plot styling"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        }) 