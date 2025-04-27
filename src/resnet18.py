import torch.nn as nn
from letters import *

# Code reference: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Stride for the convolution operation (default: 1)
        
    Returns:
        nn.Conv2d: 3x3 convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet architecture
    
    This block performs two 3x3 convolutions with batch normalization and ReLU activation.
    It implements the residual connection where the input is added to the output
    before the final activation.
    
    Attributes:
        expansion (int): Factor by which the number of output channels increases (1 for BasicBlock)
    """
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initialize the BasicBlock
        
        Args:
            inplanes: Number of input channels
            planes: Number of output channels in the first conv layer (and after expansion)
            stride: Stride for the first convolution layer (default: 1)
            downsample: Optional downsampling function for the residual connection (default: None)
        """
        super(BasicBlock, self).__init__()
        
        # First convolution block: conv -> bn -> relu
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution block: conv -> bn
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Downsample layer for residual connection when dimensions change
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        """
        Forward pass for the BasicBlock
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output after processing through the block
        """
        # Store input for residual connection
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsample to residual if needed
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture implementation
    
    A flexible implementation of ResNet that can be configured with different depths
    by specifying the number of blocks in each layer. Supports both grayscale and
    RGB inputs.
    """
    
    def __init__(self, block, layers, num_classes, grayscale):
        """
        Initialize the ResNet model
        
        Args:
            block: Block type to use (e.g., BasicBlock)
            layers: List containing number of blocks in each layer
            num_classes: Number of output classes for classification
            grayscale: Boolean indicating if input is grayscale
        """
        self.inplanes = 64
        
        # Determine input dimensions based on grayscale flag
        if grayscale:
            in_dim = 1  # Single channel for grayscale
        else:
            in_dim = 3  # Three channels for RGB
        
        super(ResNet, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Four main ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Pooling and classification layers
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize model weights using He initialization for conv layers
        and constant weights for batch normalization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for convolutional layers
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**0.5)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm weights to 1 and biases to 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create a ResNet layer composed of multiple blocks
        
        Args:
            block: Block type to use (e.g., BasicBlock)
            planes: Number of output channels for the blocks
            blocks: Number of blocks in this layer
            stride: Stride for the first block (default: 1)
            
        Returns:
            nn.Sequential: A sequence of blocks forming a layer
        """
        downsample = None
        
        # Create downsample layer if needed (when dimensions change)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        
        # First block may have stride > 1 and downsample
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        # Update inplanes for subsequent blocks
        self.inplanes = planes * block.expansion
        
        # Add remaining blocks
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the ResNet model
        
        Args:
            x: Input tensor
            
        Returns:
            tuple: (logits, probabilities)
        """
        # Initial convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Four main ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # NOTE: Average pooling is disabled as input is already 1x1 for MNIST
        # x = self.avgpool(x)
        
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        
        return logits


def ResNet18(num_classes=ABC_SIZE):
    """
    Constructs a ResNet-18 model
    
    This is a specific instantiation of ResNet with 18 layers (using BasicBlock),
    configured for grayscale images.
    
    Args:
        num_classes: Number of output classes (default: ABC_SIZE)
        
    Returns:
        ResNet: A ResNet-18 model
    """
    model = ResNet(block=BasicBlock,
                  layers=[2, 2, 2, 2],  # 4 layers with 2 blocks each = 18 layers
                  num_classes=num_classes,
                  grayscale=True)  # Set to True for grayscale images like MNIST
    return model