############
# Enhanced Neural Network Models for Surgical Skill Assessment
# Based on: Nguyen et al. (2019) "Surgical skill levels: Classification and 
# analysis using deep neural network model and motion signals"
# Computer Methods and Programs in Biomedicine 177 (2019) 1–8
############

import torch
import torch.nn as nn
import torch.nn.functional as F


class SENetBlock(nn.Module):
    """
    Squeeze-and-Excitation Network Block
    Adaptively weights each convolutional input channel
    
    Args:
        num_channels: Number of input channels
        reduction_ratio: Reduction ratio for the bottleneck (default: 8)
    """
    def __init__(self, num_channels, reduction_ratio=8):
        super(SENetBlock, self).__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio
        
        # Squeeze: Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: Two FC layers
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)
        
    def forward(self, x):
        batch_size, channels, _ = x.size()
        
        # Squeeze: Global average pooling
        squeeze = self.global_avg_pool(x).view(batch_size, channels)
        
        # Excitation: FC → ReLU → FC → Sigmoid
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        
        # Scale: multiply input by channel weights
        excitation = excitation.view(batch_size, channels, 1)
        output = x * excitation.expand_as(x)
        
        return output


class ImprovedConvLSTMModel(nn.Module):
    """
    Improved CNN-LSTM model with SENet blocks and parallel architecture
    Based on Nguyen et al. (2019) methodology
    
    Architecture:
    - Left branch: Conv1D → SENet → Conv1D → Global Avg Pooling
    - Right branch: Transpose → 2x LSTM layers
    - Concatenation → Dropout → Softmax classifier
    
    Args:
        input_channels: Number of input features/sensors
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 8)
        reduction_ratio: SENet reduction ratio (default: 8)
        dropout_rate: Dropout probability (default: 0.5)
    """
    def __init__(self, input_channels, hidden_features, output_size, 
                 lstm_hidden=8, reduction_ratio=8, dropout_rate=0.5):
        super(ImprovedConvLSTMModel, self).__init__()
        
        # LEFT BRANCH: Convolutional layers with SENet
        # First Conv1D layer: kernel_size=7, 16 filters
        self.conv1d_1 = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=16, 
            kernel_size=7, 
            stride=1, 
            padding=3  # Same padding
        )
        self.bn1 = nn.BatchNorm1d(16)
        
        # SENet block after first convolution
        self.senet = SENetBlock(num_channels=16, reduction_ratio=reduction_ratio)
        
        # Second Conv1D layer: kernel_size=5, 32 filters
        self.conv1d_2 = nn.Conv1d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=5, 
            stride=1, 
            padding=2  # Same padding
        )
        self.bn2 = nn.BatchNorm1d(32)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: LSTM layers
        # Two LSTM layers with lstm_hidden units each
        
        self.lstm = nn.LSTM(
            input_size=input_channels, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout_rate if lstm_hidden > 1 else 0
        )
        
        # FULLY CONNECTED LAYERS
        # Calculate concatenated size: 32 (from conv) + lstm_hidden (from LSTM)
        concat_size = 32 + lstm_hidden
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)
        
        # L2 regularization will be applied through weight_decay in optimizer
        
    def forward(self, x):
        """
        Forward pass through the dual-branch architecture
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_channels)
        
        Returns:
            Output logits of shape (batch_size, output_size)
        """
        # LEFT BRANCH: CNN processing
        # Transpose for Conv1D: (batch, channels, sequence)
        x_conv = torch.transpose(x, 1, 2)
        
        # First convolution with BatchNorm and ReLU
        x_conv = self.conv1d_1(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = F.relu(x_conv)
        
        # Apply SENet block
        x_conv = self.senet(x_conv)
        
        # Second convolution with BatchNorm and ReLU
        x_conv = self.conv1d_2(x_conv)
        x_conv = self.bn2(x_conv)
        x_conv = F.relu(x_conv)
        
        # Global Average Pooling: (batch, channels, sequence) → (batch, channels)
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH: LSTM processing
        # Input is already in correct format: (batch, sequence, features)
        x_lstm, _ = self.lstm(x)
        
        # Take only the last time step output
        x_lstm = x_lstm[:, -1, :]  # (batch, lstm_hidden)
        
        # CONCATENATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        
        # CLASSIFIER
        x_concat = self.dropout(x_concat)
        x_out = F.relu(self.fc1(x_concat))
        x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)
        
        return x_out


class SimplifiedConvLSTMModel(nn.Module):
    """
    SIMPLIFIED version to prevent overfitting
    
    Reductions from ImprovedConvLSTMModel:
    - Removed SENet block (reduces parameters)
    - Single Conv1D layer instead of two
    - Single LSTM layer instead of two
    - Smaller filters (8 instead of 16/32)
    - Single FC layer classifier
    - MaxPooling to reduce sequence length
    
    Args:
        input_channels: Number of input features/sensors
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layer (default: 16)
        dropout_rate: Dropout probability (default: 0.4)
    """
    def __init__(self, input_channels, hidden_features, output_size, 
                 lstm_hidden=16, dropout_rate=0.4):
        super(SimplifiedConvLSTMModel, self).__init__()
        
        # LEFT BRANCH: Simplified CNN
        self.conv1d = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=16,  # Reduced from 32
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        self.bn = nn.BatchNorm1d(16)
        self.maxpool = nn.MaxPool1d(kernel_size=2)  # Reduce sequence length
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: Single LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_channels, 
            hidden_size=lstm_hidden, 
            num_layers=1,  # Reduced from 2
            batch_first=True
        )
        
        # CLASSIFIER: Single layer
        concat_size = 16 + lstm_hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(concat_size, output_size)  # Direct to output
        
    def forward(self, x):
        # LEFT BRANCH: Simple CNN
        x_conv = torch.transpose(x, 1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = self.bn(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.maxpool(x_conv)  # Reduce overfitting
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH: Simple LSTM
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]
        
        # CONCATENATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        
        # CLASSIFIER
        x_out = self.dropout(x_concat)
        x_out = self.fc(x_out)
        
        return x_out


class EnhancedConvLSTMModel(nn.Module):
    """
    Enhanced version with residual connections and attention
    
    Additional improvements:
    - Residual connections in CNN branch
    - Attention mechanism on LSTM outputs
    - Deeper architecture option
    
    Args:
        input_channels: Number of input features/sensors
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 16)
        reduction_ratio: SENet reduction ratio (default: 8)
        dropout_rate: Dropout probability (default: 0.5)
        use_attention: Whether to use attention on LSTM outputs (default: True)
    """
    def __init__(self, input_channels, hidden_features, output_size, 
                 lstm_hidden=16, reduction_ratio=8, dropout_rate=0.5,
                 use_attention=True):
        super(EnhancedConvLSTMModel, self).__init__()
        
        self.use_attention = use_attention
        
        # LEFT BRANCH: Enhanced CNN with residual connections
        self.conv1d_1 = nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.senet1 = SENetBlock(16, reduction_ratio)
        
        self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.senet2 = SENetBlock(32, reduction_ratio)
        
        # Additional conv layer for deeper features
        self.conv1d_3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Residual connection adapter
        self.residual_adapter = nn.Conv1d(16, 64, kernel_size=1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: LSTM with optional attention
        self.lstm = nn.LSTM(
            input_size=input_channels, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout_rate
        )
        
        if self.use_attention:
            # Simple attention mechanism
            self.attention = nn.Linear(lstm_hidden, 1)
        
        # FULLY CONNECTED LAYERS
        concat_size = 64 + lstm_hidden
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.fc3 = nn.Linear(hidden_features // 2, output_size)
        
    def forward(self, x):
        # LEFT BRANCH with residual connection
        x_conv = torch.transpose(x, 1, 2)
        
        # First block
        identity = x_conv
        x_conv = F.relu(self.bn1(self.conv1d_1(x_conv)))
        x_conv = self.senet1(x_conv)
        
        # Store for residual
        residual = self.residual_adapter(x_conv)
        
        # Second block
        x_conv = F.relu(self.bn2(self.conv1d_2(x_conv)))
        x_conv = self.senet2(x_conv)
        
        # Third block with residual
        x_conv = F.relu(self.bn3(self.conv1d_3(x_conv)))
        x_conv = x_conv + residual  # Residual connection
        
        # Global pooling
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH with optional attention
        x_lstm, _ = self.lstm(x)
        
        if self.use_attention:
            # Apply attention over sequence
            attention_weights = torch.softmax(self.attention(x_lstm), dim=1)
            x_lstm = torch.sum(x_lstm * attention_weights, dim=1)
        else:
            x_lstm = x_lstm[:, -1, :]
        
        # CONCATENATION and CLASSIFICATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        
        x_out = self.dropout(x_concat)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = F.relu(self.fc2(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc3(x_out)
        
        return x_out


class RestartLearningRateScheduler:
    """
    Cosine Annealing with Warm Restarts (SGDR)
    Based on Loshchilov & Hutter (2017)
    
    Learning rate schedule that periodically restarts to escape local minima
    
    Args:
        optimizer: PyTorch optimizer
        lr_max: Maximum learning rate
        lr_min: Minimum learning rate
        T: Number of steps per cycle
        T_mult: Factor to increase cycle length after each restart (default: 1)
        decay: Decay factor for lr_max after each cycle (default: 0.9)
    """
    def __init__(self, optimizer, lr_max=0.01, lr_min=1e-5, T=3, T_mult=1, decay=0.9):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T = T
        self.T_mult = T_mult
        self.decay = decay
        
        self.current_step = 0
        self.current_cycle = 0
        self.steps_in_cycle = 0
        self.current_T = T
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        self.steps_in_cycle += 1
        
        # Calculate cosine annealing learning rate
        cos_inner = (self.steps_in_cycle / self.current_T) * 3.14159
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + torch.cos(torch.tensor(cos_inner)))
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr.item()
        
        # Check for restart
        if self.steps_in_cycle >= self.current_T:
            self.current_cycle += 1
            self.steps_in_cycle = 0
            self.lr_max *= self.decay
            self.current_T = int(self.current_T * self.T_mult)
        
        return lr.item()
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


# Example usage function
def create_improved_model(input_channels, output_size, model_type='simplified', **kwargs):
    """
    Factory function to create improved models
    
    Args:
        input_channels: Number of input features
        output_size: Number of output classes
        model_type: 'simplified', 'improved', or 'enhanced'
        **kwargs: Additional model parameters
    
    Returns:
        Neural network model
    """
    hidden_features = kwargs.get('hidden_features', 64)
    dropout_rate = kwargs.get('dropout_rate', 0.4)
    
    if model_type == 'simplified':
        lstm_hidden = kwargs.get('lstm_hidden', 16)
        model = SimplifiedConvLSTMModel(
            input_channels=input_channels,
            hidden_features=hidden_features,
            output_size=output_size,
            lstm_hidden=lstm_hidden,
            dropout_rate=dropout_rate
        )
    elif model_type == 'improved':
        lstm_hidden = kwargs.get('lstm_hidden', 8)
        reduction_ratio = kwargs.get('reduction_ratio', 8)
        model = ImprovedConvLSTMModel(
            input_channels=input_channels,
            hidden_features=hidden_features,
            output_size=output_size,
            lstm_hidden=lstm_hidden,
            reduction_ratio=reduction_ratio,
            dropout_rate=dropout_rate
        )
    elif model_type == 'enhanced':
        lstm_hidden = kwargs.get('lstm_hidden', 16)
        reduction_ratio = kwargs.get('reduction_ratio', 8)
        use_attention = kwargs.get('use_attention', True)
        model = EnhancedConvLSTMModel(
            input_channels=input_channels,
            hidden_features=hidden_features,
            output_size=output_size,
            lstm_hidden=lstm_hidden,
            reduction_ratio=reduction_ratio,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model
    
class BadmintonAdaptiveSplitModel(nn.Module):
    """
    Modelo con división adaptativa de sensores según subset del dataset
    
    División propuesta:
    - CNN Branch: Características espaciales y de movimiento
      * Body tracking (58:121) - 63 sensores: posiciones articulares
      * Insole pressure (22:58) - 36 sensores: distribución de peso
      
    - LSTM Branch: Características temporales y fisiológicas
      * Gforce/IMU (2:18) - 16 sensores: aceleraciones/orientaciones
      * Eye tracking (0:2) - 2 sensores: patrones visuales
      * Cognionics/EEG (18:22) - 4 sensores: señales cerebrales
    
    Args:
        subset: Tipo de subset de sensores ('allStreams', 'noGforce', etc.)
        output_size: Número de clases (default: 3)
        hidden_features: Tamaño de capa oculta (default: 128)
        lstm_hidden: Tamaño LSTM (default: 32)
        dropout_rate: Tasa de dropout (default: 0.4)
    """
    def __init__(self, subset='allStreams', output_size=3, 
                 hidden_features=128, lstm_hidden=32, dropout_rate=0.4):
        super(BadmintonAdaptiveSplitModel, self).__init__()
        
        self.subset = subset
        
        # Definir índices según el subset
        self.cnn_indices, self.lstm_indices = self._get_sensor_indices(subset)
        
        num_cnn_channels = len(self.cnn_indices)
        num_lstm_channels = len(self.lstm_indices)
        
        # ==========================================
        # CNN BRANCH: Spatial Movement Processing
        # ==========================================
        if num_cnn_channels > 0:
            self.use_cnn = True
            # Ajustar arquitectura según número de sensores
            if num_cnn_channels < 30:
                # Arquitectura reducida para pocos sensores
                self.conv1d_1 = nn.Conv1d(num_cnn_channels, 32, kernel_size=5, padding=2)
                self.bn1 = nn.BatchNorm1d(32)
                self.conv1d_2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm1d(64)
                cnn_output_size = 64
            else:
                # Arquitectura completa para muchos sensores
                self.conv1d_1 = nn.Conv1d(num_cnn_channels, 64, kernel_size=7, padding=3)
                self.bn1 = nn.BatchNorm1d(64)
                self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
                self.bn2 = nn.BatchNorm1d(128)
                self.conv1d_3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm1d(256)
                cnn_output_size = 256
            
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.use_cnn = False
            cnn_output_size = 0
        
        # ==========================================
        # LSTM BRANCH: Temporal Pattern Processing
        # ==========================================
        if num_lstm_channels > 0:
            self.use_lstm = True
            self.lstm = nn.LSTM(
                input_size=num_lstm_channels,
                hidden_size=lstm_hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=True
            )
            lstm_output_size = lstm_hidden * 2
        else:
            self.use_lstm = False
            lstm_output_size = 0
        
        # ==========================================
        # FUSION AND CLASSIFICATION
        # ==========================================
        concat_size = cnn_output_size + lstm_output_size
        
        if concat_size == 0:
            raise ValueError(f"No hay sensores disponibles para el subset '{subset}'")
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.fc3 = nn.Linear(hidden_features // 2, output_size)
        
    def _get_sensor_indices(self, subset):
        """
        Devuelve los índices de sensores para CNN y LSTM según el subset
        
        Estructura original:
        [0:2]    = Eye tracking
        [2:18]   = Gforce/IMU
        [18:22]  = Cognionics/EEG
        [22:58]  = Insole
        [58:121] = Body tracking
        """
        # Definir todos los índices base
        eye_indices = list(range(0, 2))
        gforce_indices = list(range(2, 18))
        cognionics_indices = list(range(18, 22))
        insole_indices = list(range(22, 58))
        body_indices = list(range(58, 121))
        
        # CNN: Body + Insole (espacial)
        # LSTM: Eye + Gforce + Cognionics (temporal/fisiológico)
        
        if subset == 'allStreams':
            cnn_sensors = insole_indices + body_indices
            lstm_sensors = eye_indices + gforce_indices + cognionics_indices
            
        elif subset == 'noGforce':
            cnn_sensors = insole_indices + body_indices
            lstm_sensors = eye_indices + cognionics_indices
            
        elif subset == 'noCognionics':
            cnn_sensors = insole_indices + body_indices
            lstm_sensors = eye_indices + gforce_indices
            
        elif subset == 'noEye':
            cnn_sensors = insole_indices + body_indices
            lstm_sensors = gforce_indices + cognionics_indices
            
        elif subset == 'noInsole':
            cnn_sensors = body_indices
            lstm_sensors = eye_indices + gforce_indices + cognionics_indices
            
        elif subset == 'noBody':
            cnn_sensors = insole_indices
            lstm_sensors = eye_indices + gforce_indices + cognionics_indices
            
        elif subset == 'onlyGforce':
            cnn_sensors = []
            lstm_sensors = gforce_indices
            
        elif subset == 'onlyEye':
            cnn_sensors = []
            lstm_sensors = eye_indices
            
        elif subset == 'onlyInsole':
            cnn_sensors = insole_indices
            lstm_sensors = []
            
        elif subset == 'onlyBody':
            cnn_sensors = body_indices
            lstm_sensors = []
            
        else:
            raise ValueError(f"Subset desconocido: {subset}")
        
        # Ajustar índices según el input real después del preprocesamiento
        # (esto depende de cómo se construye input_feature_matrices)
        cnn_adjusted, lstm_adjusted = self._adjust_indices_for_subset(
            subset, cnn_sensors, lstm_sensors
        )
        
        return cnn_adjusted, lstm_adjusted
    
    def _adjust_indices_for_subset(self, subset, cnn_sensors, lstm_sensors):
        """
        Ajusta los índices relativos al array procesado (no al original)
        """
        # Mapeo de índices originales -> índices en el array procesado
        if subset == 'allStreams':
            # No hay cambios, usar índices directos
            return cnn_sensors, lstm_sensors
        
        elif subset == 'noGforce':
            # Array: [0:2] eye + [18:121] resto
            # Ajustar índices
            new_cnn = []
            new_lstm = []
            
            for idx in cnn_sensors:
                if idx >= 18:
                    new_cnn.append(idx - 16)  # Shift por los 16 eliminados
            
            for idx in lstm_sensors:
                if idx < 2:
                    new_lstm.append(idx)
                elif idx >= 18:
                    new_lstm.append(idx - 16)
            
            return new_cnn, new_lstm
        
        elif subset == 'noCognionics':
            # Array: [0:18] eye+gforce + [22:121] resto
            new_cnn = []
            new_lstm = []
            
            for idx in cnn_sensors:
                if idx < 18:
                    new_cnn.append(idx)
                elif idx >= 22:
                    new_cnn.append(idx - 4)
            
            for idx in lstm_sensors:
                if idx < 18:
                    new_lstm.append(idx)
                elif idx >= 22:
                    new_lstm.append(idx - 4)
            
            return new_cnn, new_lstm
        
        elif subset == 'noEye':
            # Array: [2:121] todo excepto eye
            new_cnn = [idx - 2 for idx in cnn_sensors if idx >= 2]
            new_lstm = [idx - 2 for idx in lstm_sensors if idx >= 2]
            return new_cnn, new_lstm
        
        elif subset == 'noInsole':
            # Array: [0:22] + [58:121]
            new_cnn = []
            new_lstm = []
            
            for idx in cnn_sensors:
                if idx < 22:
                    new_cnn.append(idx)
                elif idx >= 58:
                    new_cnn.append(idx - 36)
            
            for idx in lstm_sensors:
                if idx < 22:
                    new_lstm.append(idx)
                elif idx >= 58:
                    new_lstm.append(idx - 36)
            
            return new_cnn, new_lstm
        
        elif subset == 'noBody':
            # Array: [0:58] solo hasta insole
            new_cnn = [idx for idx in cnn_sensors if idx < 58]
            new_lstm = [idx for idx in lstm_sensors if idx < 58]
            return new_cnn, new_lstm
        
        elif subset == 'onlyGforce':
            # Array: [2:18] -> [0:16]
            return [], list(range(0, 16))
        
        elif subset == 'onlyEye':
            # Array: [0:2]
            return [], [0, 1]
        
        elif subset == 'onlyInsole':
            # Array: [22:58] -> [0:36]
            return list(range(0, 36)), []
        
        elif subset == 'onlyBody':
            # Array: [58:121] -> [0:63]
            return list(range(0, 63)), []
        
        return cnn_sensors, lstm_sensors
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, num_sensors)
        Returns:
            logits: (batch_size, output_size)
        """
        features = []
        
        # ==========================================
        # CNN BRANCH
        # ==========================================
        if self.use_cnn and len(self.cnn_indices) > 0:
            x_spatial = x[:, :, self.cnn_indices]
            x_conv = torch.transpose(x_spatial, 1, 2)
            
            x_conv = F.relu(self.bn1(self.conv1d_1(x_conv)))
            x_conv = F.relu(self.bn2(self.conv1d_2(x_conv)))
            
            # Si hay tercera capa
            if hasattr(self, 'conv1d_3'):
                x_conv = F.relu(self.bn3(self.conv1d_3(x_conv)))
            
            x_conv = self.global_avg_pool(x_conv).squeeze(-1)
            features.append(x_conv)
        
        # ==========================================
        # LSTM BRANCH
        # ==========================================
        if self.use_lstm and len(self.lstm_indices) > 0:
            x_temporal = x[:, :, self.lstm_indices]
            x_lstm, (hn, cn) = self.lstm(x_temporal)
            x_lstm = torch.cat([hn[-2], hn[-1]], dim=1)
            features.append(x_lstm)
        
        # ==========================================
        # FUSION
        # ==========================================
        if len(features) == 0:
            raise RuntimeError("No hay características disponibles")
        
        x_fused = torch.cat(features, dim=1)
        
        # ==========================================
        # CLASSIFICATION
        # ==========================================
        x_out = self.dropout(x_fused)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = F.relu(self.fc2(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc3(x_out)
        
        return x_out



class ImprovedConvLSTMModel_RBTranspose(nn.Module):
    """
    Improved CNN-LSTM model with SENet blocks and parallel architecture
    Based on Nguyen et al. (2019) methodology with LSTM transpose for overfitting reduction
    
    Architecture:
    - Left branch: Conv1D → SENet → Conv1D → Global Avg Pooling
    - Right branch: Transpose (L×N → N×L) → 2x LSTM layers
    - Concatenation → Dropout → Softmax classifier
    
    Args:
        input_channels: Number of input features/sensors (N)
        sequence_length: Number of time steps in the sequence (L)
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 8)
        reduction_ratio: SENet reduction ratio (default: 8)
        dropout_rate: Dropout probability (default: 0.5)
    """
    def __init__(self, input_channels, sequence_length, hidden_features, output_size, 
                 lstm_hidden=8, reduction_ratio=8, dropout_rate=0.5):
        super(ImprovedConvLSTMModel_RBTranspose, self).__init__()
        
        self.input_channels = input_channels  # N
        self.sequence_length = sequence_length  # L
        
        # LEFT BRANCH: Convolutional layers with SENet
        # First Conv1D layer: kernel_size=7, 16 filters
        self.conv1d_1 = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=16, 
            kernel_size=7, 
            stride=1, 
            padding=3  # Same padding
        )
        self.bn1 = nn.BatchNorm1d(16)
        
        # SENet block after first convolution
        self.senet = SENetBlock(num_channels=16, reduction_ratio=reduction_ratio)
        
        # Second Conv1D layer: kernel_size=5, 32 filters
        self.conv1d_2 = nn.Conv1d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=5, 
            stride=1, 
            padding=2  # Same padding
        )
        self.bn2 = nn.BatchNorm1d(32)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: LSTM layers with transposed input
        # After transpose: input becomes (batch, N, L)
        # LSTM processes N time steps, each with L features
        # This reduces overfitting when L >> N (small datasets)
        self.lstm = nn.LSTM(
            input_size=sequence_length,  # Now L features per time step
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout_rate if lstm_hidden > 1 else 0
        )
        
        # FULLY CONNECTED LAYERS
        # Calculate concatenated size: 32 (from conv) + lstm_hidden (from LSTM)
        concat_size = 32 + lstm_hidden
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)
        
        # L2 regularization will be applied through weight_decay in optimizer
        
    def forward(self, x):
        """
        Forward pass through the dual-branch architecture
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_channels)
               Expected shape: (batch, L, N)
        
        Returns:
            Output logits of shape (batch_size, output_size)
        """
        # LEFT BRANCH: CNN processing
        # Transpose for Conv1D: (batch, L, N) → (batch, N, L)
        x_conv = torch.transpose(x, 1, 2)
        
        # First convolution with BatchNorm and ReLU
        x_conv = self.conv1d_1(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = F.relu(x_conv)
        
        # Apply SENet block
        x_conv = self.senet(x_conv)
        
        # Second convolution with BatchNorm and ReLU
        x_conv = self.conv1d_2(x_conv)
        x_conv = self.bn2(x_conv)
        x_conv = F.relu(x_conv)
        
        # Global Average Pooling: (batch, channels, sequence) → (batch, channels)
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH: LSTM processing with transpose
        # Transpose to reduce overfitting: (batch, L, N) → (batch, N, L)
        # This makes LSTM process N shorter sequences instead of L longer ones
        # Beneficial when L >> N and dataset is small
        x_lstm = torch.transpose(x, 1, 2)  # (batch, N, L)
        
        # LSTM processes N time steps with L features each
        x_lstm, _ = self.lstm(x_lstm)
        
        # Take only the last time step output
        x_lstm = x_lstm[:, -1, :]  # (batch, lstm_hidden)
        
        # CONCATENATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        
        # CLASSIFIER
        x_concat = self.dropout(x_concat)
        x_out = F.relu(self.fc1(x_concat))
        x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)
        
        return x_out
        
        
class EnhancedConvLSTMModel_RBTranspose(nn.Module):
    """
    Enhanced version with residual connections and attention
    
    Additional improvements:
    - Residual connections in CNN branch
    - Attention mechanism on LSTM outputs
    - Deeper architecture option
    
    Args:
        input_channels: Number of input features/sensors
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 16)
        reduction_ratio: SENet reduction ratio (default: 8)
        dropout_rate: Dropout probability (default: 0.5)
        use_attention: Whether to use attention on LSTM outputs (default: True)
    """
    def __init__(self, input_channels, hidden_features, output_size, 
                 lstm_hidden=16, reduction_ratio=8, dropout_rate=0.5,
                 use_attention=True):
        super(EnhancedConvLSTMModel_RBTranspose, self).__init__()
        
        self.use_attention = use_attention
        
        # LEFT BRANCH: Enhanced CNN with residual connections
        self.conv1d_1 = nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.senet1 = SENetBlock(16, reduction_ratio)
        
        self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.senet2 = SENetBlock(32, reduction_ratio)
        
        # Additional conv layer for deeper features
        self.conv1d_3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Residual connection adapter
        self.residual_adapter = nn.Conv1d(16, 64, kernel_size=1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: LSTM with optional attention
        self.lstm = nn.LSTM(
            input_size=input_channels, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout_rate
        )
        
        if self.use_attention:
            # Simple attention mechanism
            self.attention = nn.Linear(lstm_hidden, 1)
        
        # FULLY CONNECTED LAYERS
        concat_size = 64 + lstm_hidden
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.fc3 = nn.Linear(hidden_features // 2, output_size)
        
    def forward(self, x):
        # LEFT BRANCH with residual connection
        x_conv = torch.transpose(x, 1, 2)
        
        # First block
        identity = x_conv
        x_conv = F.relu(self.bn1(self.conv1d_1(x_conv)))
        x_conv = self.senet1(x_conv)
        
        # Store for residual
        residual = self.residual_adapter(x_conv)
        
        # Second block
        x_conv = F.relu(self.bn2(self.conv1d_2(x_conv)))
        x_conv = self.senet2(x_conv)
        
        # Third block with residual
        x_conv = F.relu(self.bn3(self.conv1d_3(x_conv)))
        x_conv = x_conv + residual  # Residual connection
        
        # Global pooling
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH with optional attention
        x_lstm, _ = self.lstm(x)
        
        if self.use_attention:
            # Apply attention over sequence
            attention_weights = torch.softmax(self.attention(x_lstm), dim=1)
            x_lstm = torch.sum(x_lstm * attention_weights, dim=1)
        else:
            x_lstm = x_lstm[:, -1, :]
        
        # CONCATENATION and CLASSIFICATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        
        x_out = self.dropout(x_concat)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = F.relu(self.fc2(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc3(x_out)
        
        return x_out

class EnhancedConvLSTMModel_RBTranspose(nn.Module):
    """
    Enhanced version with residual connections, attention, and LSTM transpose
    
    Additional improvements:
    - Residual connections in CNN branch
    - Attention mechanism on LSTM outputs
    - Deeper architecture option
    - LSTM transpose (L×N → N×L) for overfitting reduction
    
    Args:
        input_channels: Number of input features/sensors (N)
        sequence_length: Number of time steps in the sequence (L)
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 16)
        reduction_ratio: SENet reduction ratio (default: 8)
        dropout_rate: Dropout probability (default: 0.5)
        use_attention: Whether to use attention on LSTM outputs (default: True)
    """
    def __init__(self, input_channels, sequence_length, hidden_features, output_size, 
                 lstm_hidden=16, reduction_ratio=8, dropout_rate=0.5,
                 use_attention=True):
        super(EnhancedConvLSTMModel_RBTranspose, self).__init__()
        
        self.use_attention = use_attention
        self.input_channels = input_channels  # N
        self.sequence_length = sequence_length  # L
        
        # LEFT BRANCH: Enhanced CNN with residual connections
        self.conv1d_1 = nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.senet1 = SENetBlock(16, reduction_ratio)
        
        self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.senet2 = SENetBlock(32, reduction_ratio)
        
        # Additional conv layer for deeper features
        self.conv1d_3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Residual connection adapter
        self.residual_adapter = nn.Conv1d(16, 64, kernel_size=1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: LSTM with transpose for overfitting reduction
        # After transpose: (batch, L, N) → (batch, N, L)
        # LSTM processes N time steps with L features each
        self.lstm = nn.LSTM(
            input_size=sequence_length,  # L features per time step after transpose
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout_rate if lstm_hidden > 1 else 0
        )
        
        if self.use_attention:
            # Simple attention mechanism
            # Note: attention now operates over N time steps (after transpose)
            self.attention = nn.Linear(lstm_hidden, 1)
        
        # FULLY CONNECTED LAYERS
        concat_size = 64 + lstm_hidden
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.fc3 = nn.Linear(hidden_features // 2, output_size)
        
    def forward(self, x):
        """
        Forward pass through the dual-branch architecture
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_channels)
               Expected shape: (batch, L, N)
        
        Returns:
            Output logits of shape (batch_size, output_size)
        """
        # LEFT BRANCH with residual connection
        # Transpose for Conv1D: (batch, L, N) → (batch, N, L)
        x_conv = torch.transpose(x, 1, 2)
        
        # First block
        identity = x_conv
        x_conv = F.relu(self.bn1(self.conv1d_1(x_conv)))
        x_conv = self.senet1(x_conv)
        
        # Store for residual
        residual = self.residual_adapter(x_conv)
        
        # Second block
        x_conv = F.relu(self.bn2(self.conv1d_2(x_conv)))
        x_conv = self.senet2(x_conv)
        
        # Third block with residual
        x_conv = F.relu(self.bn3(self.conv1d_3(x_conv)))
        x_conv = x_conv + residual  # Residual connection
        
        # Global pooling
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH: LSTM with transpose and optional attention
        # Transpose to reduce overfitting: (batch, L, N) → (batch, N, L)
        # This makes LSTM process N shorter sequences instead of L longer ones
        # Beneficial when L >> N and dataset is small
        x_lstm = torch.transpose(x, 1, 2)  # (batch, N, L)
        
        # LSTM processes N time steps with L features each
        x_lstm, _ = self.lstm(x_lstm)
        
        if self.use_attention:
            # Apply attention over N time steps (channels after transpose)
            attention_weights = torch.softmax(self.attention(x_lstm), dim=1)
            x_lstm = torch.sum(x_lstm * attention_weights, dim=1)
        else:
            # Take last time step output (last channel)
            x_lstm = x_lstm[:, -1, :]
        
        # CONCATENATION and CLASSIFICATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        
        x_out = self.dropout(x_concat)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = F.relu(self.fc2(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc3(x_out)
        
        return x_out
        
class SimplifiedConvLSTMModel_RBTranspose(nn.Module):
    """
    SIMPLIFIED version to prevent overfitting with LSTM transpose
    
    Reductions from ImprovedConvLSTMModel:
    - Removed SENet block (reduces parameters)
    - Single Conv1D layer instead of two
    - Single LSTM layer instead of two
    - Smaller filters (16 instead of 32/64)
    - Single FC layer classifier
    - MaxPooling to reduce sequence length
    - LSTM transpose (L×N → N×L) for additional overfitting reduction
    
    Args:
        input_channels: Number of input features/sensors (N)
        sequence_length: Number of time steps in the sequence (L)
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layer (default: 16)
        dropout_rate: Dropout probability (default: 0.4)
    """
    def __init__(self, input_channels, sequence_length, hidden_features, output_size, 
                 lstm_hidden=16, dropout_rate=0.4):
        super(SimplifiedConvLSTMModel_RBTranspose, self).__init__()
        
        self.input_channels = input_channels  # N
        self.sequence_length = sequence_length  # L
        
        # LEFT BRANCH: Simplified CNN
        self.conv1d = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=16,  # Reduced filters for simplicity
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        self.bn = nn.BatchNorm1d(16)
        self.maxpool = nn.MaxPool1d(kernel_size=2)  # Reduce sequence length
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: Single LSTM layer with transpose
        # After transpose: (batch, L, N) → (batch, N, L)
        # LSTM processes N time steps with L features each
        self.lstm = nn.LSTM(
            input_size=sequence_length,  # L features per time step after transpose
            hidden_size=lstm_hidden, 
            num_layers=1,  # Single layer for simplicity
            batch_first=True
        )
        
        # CLASSIFIER: Single layer (direct to output)
        concat_size = 16 + lstm_hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(concat_size, output_size)  # Direct to output
        
    def forward(self, x):
        """
        Forward pass through the simplified dual-branch architecture
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_channels)
               Expected shape: (batch, L, N)
        
        Returns:
            Output logits of shape (batch_size, output_size)
        """
        # LEFT BRANCH: Simple CNN
        # Transpose for Conv1D: (batch, L, N) → (batch, N, L)
        x_conv = torch.transpose(x, 1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = self.bn(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.maxpool(x_conv)  # Reduce overfitting by downsampling
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH: Simple LSTM with transpose
        # Transpose to reduce overfitting: (batch, L, N) → (batch, N, L)
        # This makes LSTM process N shorter sequences instead of L longer ones
        # Particularly effective when L >> N and dataset is small
        x_lstm = torch.transpose(x, 1, 2)  # (batch, N, L)
        
        # LSTM processes N time steps with L features each
        x_lstm, _ = self.lstm(x_lstm)
        
        # Take last time step output (last of N time steps)
        x_lstm = x_lstm[:, -1, :]  # (batch, lstm_hidden)
        
        # CONCATENATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        
        # CLASSIFIER: Single layer with dropout
        x_out = self.dropout(x_concat)
        x_out = self.fc(x_out)
        
        return x_out
        
class EnhancedConvLSTMModelv2Medium(nn.Module):
    """
    MEDIUM complexity version - Reduced overfitting risk
    
    Reductions from v2 full version:
    - Removed one SENet block (kept only after first conv)
    - Removed third convolutional layer
    - Removed residual connection (simpler architecture)
    - Reduced to 2 FC layers instead of 3
    - Smaller filter sizes (16 → 32 instead of 16 → 32 → 64)
    
    Args:
        input_channels: Number of input features/sensors (N)
        sequence_length: Number of time steps in the sequence (L)
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 16)
        reduction_ratio: SENet reduction ratio (default: 8)
        dropout_rate: Dropout probability (default: 0.5)
        use_attention: Whether to use attention on LSTM outputs (default: True)
    """
    def __init__(self, input_channels, sequence_length, hidden_features, output_size, 
                 lstm_hidden=16, reduction_ratio=8, dropout_rate=0.5,
                 use_attention=True):
        super(EnhancedConvLSTMModelv2Medium, self).__init__()
        
        self.use_attention = use_attention
        self.input_channels = input_channels  # N
        self.sequence_length = sequence_length  # L
        
        # LEFT BRANCH: Simplified CNN (2 conv layers, 1 SENet)
        self.conv1d_1 = nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.senet = SENetBlock(16, reduction_ratio)  # Only one SENet
        
        self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: LSTM with transpose
        self.lstm = nn.LSTM(
            input_size=sequence_length,  # L features per time step after transpose
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout_rate if lstm_hidden > 1 else 0
        )
        
        if self.use_attention:
            self.attention = nn.Linear(lstm_hidden, 1)
        
        # FULLY CONNECTED LAYERS (2 layers instead of 3)
        concat_size = 32 + lstm_hidden  # 32 from conv, lstm_hidden from LSTM
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, L, N)
        Returns:
            Output logits of shape (batch, output_size)
        """
        # LEFT BRANCH: Simplified CNN
        x_conv = torch.transpose(x, 1, 2)  # (batch, L, N) → (batch, N, L)
        
        # First conv block with SENet
        x_conv = F.relu(self.bn1(self.conv1d_1(x_conv)))
        x_conv = self.senet(x_conv)
        
        # Second conv block (no SENet)
        x_conv = F.relu(self.bn2(self.conv1d_2(x_conv)))
        
        # Global pooling
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH: LSTM with transpose
        x_lstm = torch.transpose(x, 1, 2)  # (batch, L, N) → (batch, N, L)
        x_lstm, _ = self.lstm(x_lstm)
        
        if self.use_attention:
            attention_weights = torch.softmax(self.attention(x_lstm), dim=1)
            x_lstm = torch.sum(x_lstm * attention_weights, dim=1)
        else:
            x_lstm = x_lstm[:, -1, :]
        
        # CONCATENATION and CLASSIFICATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        
        x_out = self.dropout(x_concat)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)
        
        return x_out


class EnhancedConvLSTMModelv2Light(nn.Module):
    """
    LIGHT complexity version - Minimal overfitting risk
    
    Maximum reductions from v2 full version:
    - Removed ALL SENet blocks
    - Single convolutional layer
    - Single LSTM layer
    - Single FC layer (direct classification)
    - Removed attention mechanism option
    - Smallest filter size (16 filters only)
    
    Args:
        input_channels: Number of input features/sensors (N)
        sequence_length: Number of time steps in the sequence (L)
        hidden_features: Hidden layer size (not used, kept for API consistency)
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layer (default: 16)
        dropout_rate: Dropout probability (default: 0.4)
    """
    def __init__(self, input_channels, sequence_length, hidden_features, output_size, 
                 lstm_hidden=16, dropout_rate=0.4):
        super(EnhancedConvLSTMModelv2Light, self).__init__()
        super(EnhancedConvLSTMModelv2Light, self).__init__()
        
        self.input_channels = input_channels  # N
        self.sequence_length = sequence_length  # L
        
        # LEFT BRANCH: Minimal CNN (single conv layer)
        self.conv1d = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm1d(16)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: Single LSTM with transpose
        self.lstm = nn.LSTM(
            input_size=sequence_length,  # L features per time step after transpose
            hidden_size=lstm_hidden, 
            num_layers=1,  # Single layer
            batch_first=True
        )
        
        # FULLY CONNECTED: Direct classification (single layer)
        concat_size = 16 + lstm_hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(concat_size, output_size)  # Direct to output
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, L, N)
        Returns:
            Output logits of shape (batch, output_size)
        """
        # LEFT BRANCH: Minimal CNN
        x_conv = torch.transpose(x, 1, 2)  # (batch, L, N) → (batch, N, L)
        x_conv = F.relu(self.bn(self.conv1d(x_conv)))
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH: Single LSTM with transpose
        x_lstm = torch.transpose(x, 1, 2)  # (batch, L, N) → (batch, N, L)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm[:, -1, :]  # Last time step only
        
        # CONCATENATION and CLASSIFICATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        x_out = self.dropout(x_concat)
        x_out = self.fc(x_out)
        
        return x_out     

class EnhancedConvLSTMModelv2Minimal(nn.Module):
    """
    MINIMAL complexity version - Maximum overfitting protection
    
    Ultra-simplified architecture:
    - Removed BatchNorm (less parameters)
    - Reduced filters to 8 (smallest viable)
    - Reduced LSTM hidden to 8
    - Removed Global Average Pooling (use MaxPool instead)
    - Minimal dropout (0.3)
    
    Args:
        input_channels: Number of input features/sensors (N)
        sequence_length: Number of time steps in the sequence (L)
        hidden_features: Hidden layer size (not used, kept for API consistency)
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layer (default: 8)
        dropout_rate: Dropout probability (default: 0.3)
    """
    def __init__(self, input_channels, sequence_length, hidden_features, output_size, 
                 lstm_hidden=8, dropout_rate=0.3):
        super(EnhancedConvLSTMModelv2Minimal, self).__init__()
        
        self.input_channels = input_channels  # N
        self.sequence_length = sequence_length  # L
        
        # LEFT BRANCH: Ultra-minimal CNN
        self.conv1d = nn.Conv1d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)  # Aggressive downsampling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # RIGHT BRANCH: Minimal LSTM with transpose
        self.lstm = nn.LSTM(
            input_size=sequence_length,  # L features per time step after transpose
            hidden_size=lstm_hidden, 
            num_layers=1,
            batch_first=True
        )
        
        # FULLY CONNECTED: Direct classification
        concat_size = 8 + lstm_hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(concat_size, output_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, L, N)
        Returns:
            Output logits of shape (batch, output_size)
        """
        # LEFT BRANCH: Ultra-minimal CNN
        x_conv = torch.transpose(x, 1, 2)  # (batch, L, N) → (batch, N, L)
        x_conv = F.relu(self.conv1d(x_conv))
        x_conv = self.maxpool(x_conv)  # Aggressive downsampling
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # RIGHT BRANCH: Minimal LSTM with transpose
        x_lstm = torch.transpose(x, 1, 2)  # (batch, L, N) → (batch, N, L)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm[:, -1, :]
        
        # CONCATENATION and CLASSIFICATION
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        x_out = self.dropout(x_concat)
        x_out = self.fc(x_out)
        
        return x_out        