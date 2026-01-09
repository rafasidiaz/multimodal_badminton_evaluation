"""
Integration script to use improved models with existing training pipeline
Add this to your notebook or import these functions
"""

import torch
import torch.optim as optim
import torch.nn as nn
from ImprovedSurgicalSkillModels import (
    ImprovedConvLSTMModel, 
    EnhancedConvLSTMModel,
    RestartLearningRateScheduler
)


def get_improved_model(modelName, input_channels, hidden_features, output_size, device):
    """
    Enhanced model selection function to replace the existing one in your notebook
    
    Args:
        modelName: Name of the model ('ImprovedConvLSTM', 'EnhancedConvLSTM', or existing models)
        input_channels: Number of input features
        hidden_features: Hidden layer size
        output_size: Number of output classes
        device: torch device
    
    Returns:
        model: The initialized model
        train_model: Boolean indicating if model should be trained
    """
    
    if modelName == "ImprovedConvLSTM":
        model = ImprovedConvLSTMModel(
            input_channels=input_channels,
            hidden_features=hidden_features,
            output_size=output_size,
            lstm_hidden=8,
            reduction_ratio=8,
            dropout_rate=0.5
        ).to(device)
        train_model = True
        
    elif modelName == "EnhancedConvLSTM":
        model = EnhancedConvLSTMModel(
            input_channels=input_channels,
            hidden_features=hidden_features,
            output_size=output_size,
            lstm_hidden=16,
            reduction_ratio=8,
            dropout_rate=0.5,
            use_attention=True
        ).to(device)
        train_model = True
        
    else:
        # Fall back to existing models
        from SciDataModels import (
            Conv1DRefinedModel, 
            ConvLSTMRefinedModel, 
            LSTMRefinedModel, 
            TransformerRefinedModel, 
            MajorityClassBaseline
        )
        
        if modelName == "Conv1D":
            model = Conv1DRefinedModel(input_channels, hidden_features, output_size).to(device)
            train_model = True
        elif modelName == "LSTM":
            model = LSTMRefinedModel(input_channels, hidden_features, output_size).to(device)
            train_model = True
        elif modelName == "ConvLSTM":
            model = ConvLSTMRefinedModel(input_channels, hidden_features, output_size).to(device)
            train_model = True
        elif modelName == "Transformer":
            model = TransformerRefinedModel(input_channels, hidden_features, output_size).to(device)
            train_model = True
        elif modelName == "Baseline":
            # Note: majority_class needs to be computed from train_dataset
            model = None  # Will be created later with actual data
            train_model = False
        else:
            raise ValueError(f"Unknown model name: {modelName}")
    
    return model, train_model


def get_optimizer_and_scheduler(model, learning_rate, use_restart_lr=True):
    """
    Create optimizer and learning rate scheduler
    
    Args:
        model: PyTorch model
        learning_rate: Initial learning rate
        use_restart_lr: Whether to use restart learning rate schedule
    
    Returns:
        optimizer: Adam optimizer
        scheduler: Learning rate scheduler (or None)
    """
    # Adam optimizer with L2 regularization (weight_decay)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4  # L2 regularization
    )
    
    if use_restart_lr:
        # Restart learning rate scheduler as described in paper
        scheduler = RestartLearningRateScheduler(
            optimizer=optimizer,
            lr_max=learning_rate,
            lr_min=learning_rate / 100,
            T=3,  # Restart every 3 epochs
            T_mult=1,
            decay=0.9
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_epoch_with_scheduler(model, train_loader, criterion, optimizer, scheduler, device):
    """
    Training loop for one epoch with optional scheduler
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (can be None)
        device: torch device
    
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
        true_labels: List of true labels
        predictions: List of predictions
    """
    model.train()
    running_loss = 0.0
    correct_instance = 0
    total_instance = 0
    
    train_true_labels = []
    train_predictions = []
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_instance += targets.size(0)
        correct_instance += (predicted == targets).sum().item()
        
        # Store for metrics
        running_loss += loss.item()
        train_true_labels.extend(targets.cpu().numpy())
        train_predictions.extend(predicted.cpu().detach().numpy())
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_instance / total_instance
    
    return avg_loss, accuracy, train_true_labels, train_predictions


# Example configuration for your notebook
IMPROVED_MODEL_CONFIG = {
    'ImprovedConvLSTM': {
        'hidden_features': [32, 64],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [128],
        'use_restart_lr': True,
        'description': 'CNN-LSTM with SENet blocks (paper architecture)'
    },
    'EnhancedConvLSTM': {
        'hidden_features': [64, 128],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [128],
        'use_restart_lr': True,
        'description': 'Enhanced CNN-LSTM with attention and residual connections'
    }
}


def print_model_summary(model, input_shape=(1, 150, 121)):
    """
    Print model summary with parameter count
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape for testing
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*60}\n")
    
    # Test forward pass
    try:
        x = torch.randn(input_shape)
        with torch.no_grad():
            output = model(x)
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output.shape}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Error in forward pass test: {e}\n")


# Integration snippet for your notebook
NOTEBOOK_INTEGRATION_CODE = """
# Add to your model_list in the notebook:
model_list = [
    'ImprovedConvLSTM',  # Paper architecture with SENet
    'EnhancedConvLSTM',  # Enhanced version with attention
    'ConvLSTM',          # Original for comparison
    'LSTM',
    'Conv1D',
    'Transformer'
]

# In your training loop, replace model initialization with:
if modelName in ['ImprovedConvLSTM', 'EnhancedConvLSTM']:
    model, train_model = get_improved_model(
        modelName, 
        len(input_feature_matrices[0,0,:]), 
        hidden_features, 
        label_num, 
        device
    )
    
    # Use improved optimizer and scheduler
    criterion_cf = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, 
        learning_rate, 
        use_restart_lr=True
    )
    
    # Print model summary
    print_model_summary(model, input_shape=(1, 150, len(input_feature_matrices[0,0,:])))
    
    # In training loop, use:
    for epoch in range(num_epochs):
        avg_loss, accuracy, true_labels, predictions = train_epoch_with_scheduler(
            model, train_loader, criterion_cf, optimizer, scheduler, device
        )
        # ... rest of your training code ...
"""


if __name__ == "__main__":
    print("Improved Models Integration Script")
    print("="*60)
    print("\nAvailable improved models:")
    for model_name, config in IMPROVED_MODEL_CONFIG.items():
        print(f"\n{model_name}:")
        print(f"  Description: {config['description']}")
        print(f"  Suggested hidden_features: {config['hidden_features']}")
        print(f"  Suggested learning_rate: {config['learning_rate']}")
        print(f"  Use restart LR: {config['use_restart_lr']}")
    
    print("\n" + "="*60)
    print("\nTo integrate with your notebook:")
    print(NOTEBOOK_INTEGRATION_CODE)
