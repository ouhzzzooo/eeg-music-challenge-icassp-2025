import torch
import torch.nn as nn
import math

class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=None, dilation=1):
        super().__init__()
        if padding is None:
            padding = (kernel[0] // 2, kernel[1] // 2)
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=padding, dilation=dilation, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, kernel_size, stride, dilation_list, in_size):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (n_layers - len(dilation_list))

        # Calculate padding for each temporal layer to keep consistent output sizes
        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_size, stride, 
                padding=(0, (kernel_size[1] // 2) * dilation[1]),  # Adaptive padding to ensure even sizes
                dilation=dilation
            ) for dilation in dilation_list
        ])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.layers):
            out = layer(x)
            #print(f"[TemporalBlock] Layer {i} output shape: {out.shape}")
            features.append(out)

        # Ensure consistent concatenation
        min_width = min(feature.shape[3] for feature in features)
        features = [feature[:, :, :, :min_width] for feature in features]  # Crop to minimum width if needed

        out = torch.cat(features, 1)
        return out

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spatial_layers, stride, input_height):
        super().__init__()
       
        kernel_list = []
        for i in range(num_spatial_layers):
            kernel_height = min(input_height // (i + 1), 3)  # Ensure kernel height is at most 3
            kernel_list.append((kernel_height, 1))

        padding = []
        for kernel in kernel_list:
            temp_pad = math.floor((kernel[0] - 1) / 2)
            padding.append((temp_pad, 0))

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_list[i], stride, padding[i], 1
            ) for i in range(num_spatial_layers)
        ])

    def forward(self, x):
        features = []
        for layer in self.layers:
            out = layer(x)
            features.append(out)
        out = torch.cat(features, 1)
        return out

def conv3x3(in_channels, out_channels, stride=1, input_size=(3, 3)):
    kernel_size = (min(input_size[0], 3), min(input_size[1], 3))  # Cap kernel size at (3,3)
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        #print(f"[ResidualBlock] Input shape: {x.shape}")
        residual = x
        out = self.conv1(x)
        #print(f"[ResidualBlock] After conv1 shape: {out.shape}")
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #print(f"[ResidualBlock] After conv2 shape: {out.shape}")
        out = self.bn2(out)
        
        if self.downsample:
            residual = self.downsample(x)
            #print(f"[ResidualBlock] After downsample shape: {residual.shape}")
            
        out += residual
        out = self.relu(out)
        #print(f"[ResidualBlock] Output shape: {out.shape}")
        return out