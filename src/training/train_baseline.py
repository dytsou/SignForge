import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.dataset import GTSRBDataset
from core import utils
from visualization.visualizer import TrainingVisualizer
from core.models import SimpleCNN, ModelProfiler

def main(epochs=20, batch=128, lr=1e-3):
    """
    Train a baseline CNN model with visualization

    Args:
        epochs (int): Number of epochs to train
        batch (int): Batch size
        lr (float): Learning rate
    """
    root = Path(__file__).resolve().parents[2]
    utils.seed_everything()
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(save_dir=root/"results/visualizations/baseline")
    
    train_ds = GTSRBDataset(root/"data/annotations/train.json", root, True)
    test_ds  = GTSRBDataset(root/"data/annotations/test.json",  root, False)
    train_loader = DataLoader(train_ds, batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch, shuffle=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = SimpleCNN().to(device)
    
    # Profile the model
    model_profile = ModelProfiler.get_model_profile(model, device=device)
    visualizer.set_model_profile(model_profile)
    
    loss_fn, optim_ = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr)
    
    print("Starting Baseline CNN Training with Visualization...")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch Size: {batch}, Learning Rate: {lr}")
    print(f"Model Parameters: {model_profile['parameters_millions']:.2f}M")
    print(f"Model Size: {model_profile['model_size_mb']:.2f} MB")
    
    for ep in range(1, epochs+1):
        # Train and get average loss and timing
        avg_loss, epoch_time = utils.train_one_epoch(model, train_loader, loss_fn, optim_, device, ep)
        
        # Evaluate and get accuracy
        acc = utils.evaluate(model, test_loader, device)
        
        # Log metrics for visualization
        visualizer.log_metrics(ep, avg_loss, acc, epoch_time)
        
        # Save checkpoint
        utils.save_checkpoint(model, root/f"models/baseline_ep{ep}.pth")
        
        visualizer.plot_training_curves()
    
    # Generate comprehensive training report
    class_names = utils.get_class_names()
    visualizer.generate_training_report(model, test_loader, device, "baseline_cnn", class_names)

if __name__ == "__main__":
    main() 