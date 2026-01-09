import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """
    Residual Block for 1D temporal data
    Implements skip connections to enable deeper networks
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 3)
        stride: Stride for convolution (default: 1)
        downsample: Optional downsample layer for skip connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNetLSTMModel(nn.Module):
    """
    ResNet-inspired architecture with LSTM branch
    Uses residual blocks to enable deeper feature extraction
    
    Args:
        input_channels: Number of input features/sensors
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 16)
        num_blocks: Number of residual blocks (default: 3)
        dropout_rate: Dropout probability (default: 0.4)
    """
    def __init__(self, input_channels, hidden_features, output_size, 
                 lstm_hidden=16, num_blocks=3, dropout_rate=0.4):
        super(ResNetLSTMModel, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Residual blocks
        self.layer1 = self._make_layer(32, 32, num_blocks)
        self.layer2 = self._make_layer(32, 64, num_blocks, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Classifier
        concat_size = 64 + lstm_hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # CNN branch
        x_conv = torch.transpose(x, 1, 2)
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))
        x_conv = self.layer1(x_conv)
        x_conv = self.layer2(x_conv)
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # LSTM branch
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]
        
        # Concatenation and classification
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        x_out = self.dropout(x_concat)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)
        
        return x_out


class InceptionModule(nn.Module):
    """
    Inception module for multi-scale temporal feature extraction
    Processes input at different temporal scales simultaneously
    
    Args:
        in_channels: Number of input channels
        out_1x1: Output channels for 1x1 conv
        reduce_3x3: Reduction channels before 3x3 conv
        out_3x3: Output channels for 3x3 conv
        reduce_5x5: Reduction channels before 5x5 conv
        out_5x5: Output channels for 5x5 conv
        pool_proj: Output channels for pooling projection
    """
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, 
                 reduce_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm1d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, reduce_3x3, kernel_size=1),
            nn.BatchNorm1d(reduce_3x3),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_3x3),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, reduce_5x5, kernel_size=1),
            nn.BatchNorm1d(reduce_5x5),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        # Pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm1d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionLSTMModel(nn.Module):
    """
    Inception-inspired architecture with LSTM
    Multi-scale feature extraction for temporal patterns
    
    Args:
        input_channels: Number of input features/sensors
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 16)
        dropout_rate: Dropout probability (default: 0.4)
    """
    def __init__(self, input_channels, hidden_features, output_size, 
                 lstm_hidden=16, dropout_rate=0.4):
        super(InceptionLSTMModel, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Inception modules
        self.inception1 = InceptionModule(32, 16, 16, 32, 8, 16, 16)  # Output: 80 channels
        self.inception2 = InceptionModule(80, 32, 32, 64, 16, 32, 32)  # Output: 160 channels
        
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Classifier
        concat_size = 160 + lstm_hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)
    
    def forward(self, x):
        # CNN branch with Inception modules
        x_conv = torch.transpose(x, 1, 2)
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))
        x_conv = self.inception1(x_conv)
        x_conv = self.maxpool(x_conv)
        x_conv = self.inception2(x_conv)
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # LSTM branch
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]
        
        # Concatenation and classification
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        x_out = self.dropout(x_concat)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)
        
        return x_out


class DenseBlock(nn.Module):
    """
    Dense block for DenseNet architecture
    Each layer receives feature maps from all preceding layers
    
    Args:
        in_channels: Number of input channels
        growth_rate: Number of output channels per layer
        num_layers: Number of layers in dense block
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(layer_in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(layer_in_channels, growth_rate, kernel_size=3, padding=1)
                )
            )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNetLSTMModel(nn.Module):
    """
    DenseNet-inspired architecture with LSTM
    Dense connections for efficient feature reuse
    
    Args:
        input_channels: Number of input features/sensors
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 16)
        growth_rate: Growth rate for dense blocks (default: 12)
        dropout_rate: Dropout probability (default: 0.4)
    """
    def __init__(self, input_channels, hidden_features, output_size, 
                 lstm_hidden=16, growth_rate=12, dropout_rate=0.4):
        super(DenseNetLSTMModel, self).__init__()
        
        # Initial convolution
        num_init_features = 32
        self.conv1 = nn.Conv1d(input_channels, num_init_features, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(num_init_features)
        
        # Dense blocks
        num_layers_per_block = 4
        self.dense1 = DenseBlock(num_init_features, growth_rate, num_layers_per_block)
        num_features = num_init_features + num_layers_per_block * growth_rate
        
        # Transition layer
        self.transition = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_features, num_features // 2, kernel_size=1),
            nn.AvgPool1d(kernel_size=2)
        )
        num_features = num_features // 2
        
        self.dense2 = DenseBlock(num_features, growth_rate, num_layers_per_block)
        num_features += num_layers_per_block * growth_rate
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Classifier
        concat_size = num_features + lstm_hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(concat_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)
    
    def forward(self, x):
        # CNN branch with Dense blocks
        x_conv = torch.transpose(x, 1, 2)
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))
        x_conv = self.dense1(x_conv)
        x_conv = self.transition(x_conv)
        x_conv = self.dense2(x_conv)
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # LSTM branch
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]
        
        # Concatenation and classification
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        x_out = self.dropout(x_concat)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)
        
        return x_out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) block
    Uses dilated causal convolutions for long-range dependencies
    
    Args:
        num_inputs: Number of input channels
        num_channels: List of channels for each layer
        kernel_size: Convolution kernel size (default: 3)
        dropout: Dropout rate (default: 0.2)
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=padding, dilation=dilation_size),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    """
    Pure Temporal Convolutional Network model
    Alternative to LSTM for temporal modeling with parallel computation
    
    Args:
        input_channels: Number of input features/sensors
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        tcn_channels: List of channel sizes for TCN layers
        kernel_size: TCN kernel size (default: 3)
        dropout_rate: Dropout probability (default: 0.3)
    """
    def __init__(self, input_channels, hidden_features, output_size,
                 tcn_channels=[32, 64, 128], kernel_size=3, dropout_rate=0.3):
        super(TCNModel, self).__init__()
        
        self.tcn = TemporalConvNet(input_channels, tcn_channels, 
                                   kernel_size=kernel_size, dropout=dropout_rate)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(tcn_channels[-1], hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)
    
    def forward(self, x):
        # TCN expects (batch, channels, sequence)
        x = torch.transpose(x, 1, 2)
        x = self.tcn(x)
        x = self.global_avg_pool(x).squeeze(-1)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MobileNetBlock(nn.Module):
    """
    Depthwise Separable Convolution block (MobileNet-inspired)
    Efficient architecture with fewer parameters
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for depthwise convolution
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileNetBlock, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, 
                     stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetLSTMModel(nn.Module):
    """
    MobileNet-inspired lightweight architecture with LSTM
    Efficient model with depthwise separable convolutions
    
    Args:
        input_channels: Number of input features/sensors
        hidden_features: Hidden layer size for final classification
        output_size: Number of classes
        lstm_hidden: Hidden size for LSTM layers (default: 16)
        dropout_rate: Dropout probability (default: 0.3)
    """
    def __init__(self, input_channels, hidden_features, output_size,
                 lstm_hidden=16, dropout_rate=0.3):
        super(MobileNetLSTMModel, self).__init__()
        
        # Initial standard convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise separable blocks
        self.mobile1 = MobileNetBlock(32, 64, stride=1)
        self.mobile2 = MobileNetBlock(64, 128, stride=2)
        self.mobile3 = MobileNetBlock(128, 128, stride=1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Classifier
        concat_size = 128 + lstm_hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(concat_size, output_size)
    
    def forward(self, x):
        # CNN branch
        x_conv = torch.transpose(x, 1, 2)
        x_conv = self.conv1(x_conv)
        x_conv = self.mobile1(x_conv)
        x_conv = self.mobile2(x_conv)
        x_conv = self.mobile3(x_conv)
        x_conv = self.global_avg_pool(x_conv).squeeze(-1)
        
        # LSTM branch
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]
        
        # Concatenation and classification
        x_concat = torch.cat([x_conv, x_lstm], dim=1)
        x_out = self.dropout(x_concat)
        x_out = self.fc(x_out)
        
        return x_out