import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.dataset import GTSRBDataset
from core import utils
from visualization.visualizer import TrainingVisualizer
from core.models import ModelProfiler

def main(epochs=15, batch=128, lr=5e-4, freeze_ratio=0.7):
    """
    Train a MobileNetV2 model with visualization

    Args:
        epochs (int): Number of epochs to train
        batch (int): Batch size
        lr (float): Learning rate
    """
    root = Path(__file__).resolve().parents[2]
    utils.seed_everything()
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(save_dir=root/"results/visualizations/mobilenet")
    
    train_ds = GTSRBDataset(root/"data/annotations/train.json", root, True)
    test_ds  = GTSRBDataset(root/"data/annotations/test.json",  root, False)
    train_loader = DataLoader(train_ds, batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch, shuffle=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, 43)
    model.to(device)
    
    # Freeze the first freeze_ratio percentage of layers
    num_feats = len(model.features)
    for i, p in enumerate(model.features.parameters()):
        if i / num_feats < freeze_ratio:
            p.requires_grad_(False)
            
    # Profile the model
    model_profile = ModelProfiler.get_model_profile(model, device=device)
    visualizer.set_model_profile(model_profile)
    
    optim_ = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr)
    loss_fn = nn.CrossEntropyLoss()
    
    print("Starting MobileNetV2 Training with Visualization...")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch Size: {batch}, Learning Rate: {lr}")
    print(f"Freeze Ratio: {freeze_ratio}")
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
        utils.save_checkpoint(model, root/f"models/mobilenet_ep{ep}.pth")
        
        visualizer.plot_training_curves()
    
    # Generate comprehensive training report
    class_names = utils.get_class_names()
    visualizer.generate_training_report(model, test_loader, device, "mobilenet_v2", class_names)

if __name__ == "__main__":
    main() 