import torch, random, numpy as np
from tqdm import tqdm
import time

def seed_everything(seed=42):
    """Seed everything for reproducibility"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def train_one_epoch(model, loader, loss_fn, optim, device, epoch, visualizer=None):
    """Enhanced training function with loss tracking for visualization"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    epoch_start_time = time.time()
    
    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", unit="batch")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / num_batches
    
    return avg_loss, epoch_time

@torch.no_grad()
def evaluate(model, loader, device, desc="[Val]"):
    """Evaluate model accuracy"""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    acc = 100.0 * correct / total
    print(f"{desc} Acc = {acc:.2f}%")
    return acc

@torch.no_grad()
def evaluate_detailed(model, loader, device, class_names=None):
    """Enhanced evaluation with detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    correct = total = 0
    
    for x, y in tqdm(loader, desc="[Evaluation]"):
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    acc = 100.0 * correct / total
    print(f"Overall Accuracy: {acc:.2f}%")
    
    # Calculate per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    if class_names:
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return acc, all_preds, all_labels

def train_model_generic(model, train_loader, test_loader, device, model_name, 
                       epochs=15, lr=1e-3, optimizer_type='adam', 
                       visualizer=None, model_profile=None):
    """Generic training function for ablation studies"""
    
    # Setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Set model profile if provided
    if visualizer and model_profile:
        visualizer.set_model_profile(model_profile)
    
    print(f"Training {model_name}...")
    print(f"Epochs: {epochs}, LR: {lr}, Optimizer: {optimizer_type}")
    if model_profile:
        print(f"Parameters: {model_profile.get('parameters_millions', 0):.2f}M")
        print(f"Model Size: {model_profile.get('model_size_mb', 0):.2f} MB")
    
    for epoch in range(1, epochs + 1):
        # Train one epoch
        avg_loss, epoch_time = train_one_epoch(model, train_loader, loss_fn, 
                                              optimizer, device, epoch)
        
        # Evaluate
        acc = evaluate(model, test_loader, device)
        
        # Log metrics
        if visualizer:
            visualizer.log_metrics(epoch, avg_loss, acc, epoch_time)
            
            # Plot curves periodically
            if epoch % max(1, epochs // 3) == 0:
                visualizer.plot_training_curves()
    
    return model, visualizer

def save_checkpoint(model, path):
    """Save model checkpoint"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_checkpoint(model, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    return model

def get_class_names():
    """Return GTSRB class names for visualization"""
    # German Traffic Sign Recognition Benchmark class names
    class_names = [
        'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
        'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
        'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
        'No passing', 'No passing for vehicles over 3.5 metric tons', 
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 
        'Keep left', 'Roundabout mandatory', 'End of no passing', 
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names
