import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, efficientnet_b0, vgg11, vgg16
import time
from pathlib import Path

class SimpleCNN(nn.Module):
    """Baseline simple CNN model"""
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """Forward pass"""
        x = self.features(x).flatten(1)
        return self.classifier(x)

class LightCNN(nn.Module):
    """Ultra-lightweight CNN for speed comparison"""
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        """Forward pass"""
        x = self.features(x).flatten(1)
        return self.classifier(x)

class DeepCNN(nn.Module):
    """Deeper CNN for comparing depth impact"""
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2  
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass"""
        x = self.features(x).flatten(1)
        return self.classifier(x)

class WideCNN(nn.Module):
    """Wider CNN for comparing width impact"""
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """Forward pass"""
        x = self.features(x).flatten(1)
        return self.classifier(x)

def create_resnet18(num_classes=43, pretrained=True):
    """ResNet18 model"""
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_resnet34(num_classes=43, pretrained=True):
    """ResNet34 model"""
    model = resnet34(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_efficientnet_b0(num_classes=43, pretrained=True):
    """EfficientNet-B0 model"""
    model = efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def create_vgg11(num_classes=43, pretrained=True):
    """VGG11 model"""
    model = vgg11(weights="IMAGENET1K_V1" if pretrained else None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

def create_vgg16(num_classes=43, pretrained=True):
    """VGG16 model"""
    model = vgg16(weights="IMAGENET1K_V1" if pretrained else None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

class ModelProfiler:
    """Utility class to profile model characteristics"""
    
    @staticmethod
    def count_parameters(model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    @staticmethod
    def estimate_model_size(model):
        """Estimate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    @staticmethod
    def measure_inference_time(model, input_shape=(1, 3, 32, 32), device='cpu', num_runs=100):
        """Measure average inference time"""
        model.eval()
        dummy_input = torch.randn(*input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / num_runs
        return avg_time_ms
    
    @staticmethod
    def get_model_profile(model, input_shape=(1, 3, 32, 32), device='cpu'):
        """Get comprehensive model profile"""
        total_params, trainable_params = ModelProfiler.count_parameters(model)
        model_size_mb = ModelProfiler.estimate_model_size(model)
        inference_time = ModelProfiler.measure_inference_time(model, input_shape, device)
        
        profile = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'inference_time_ms': inference_time,
            'parameters_millions': total_params / 1e6
        }
        return profile

# Model registry for easy access
MODEL_REGISTRY = {
    'simple_cnn': SimpleCNN,
    'light_cnn': LightCNN,
    'deep_cnn': DeepCNN,
    'wide_cnn': WideCNN,
    'resnet18': lambda: create_resnet18(pretrained=True),
    'resnet18_scratch': lambda: create_resnet18(pretrained=False),
    'resnet34': lambda: create_resnet34(pretrained=True),
    'efficientnet_b0': lambda: create_efficientnet_b0(pretrained=True),
    'efficientnet_b0_scratch': lambda: create_efficientnet_b0(pretrained=False),
    'vgg11': lambda: create_vgg11(pretrained=True),
    'vgg16': lambda: create_vgg16(pretrained=True),
}

def get_model(model_name, **kwargs):
    """Get model by name from registry"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_fn = MODEL_REGISTRY[model_name]
    return model_fn(**kwargs) if kwargs else model_fn() 