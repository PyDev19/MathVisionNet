import math
import torch
from torch import Tensor
from torch.nn import Module, Parameter, Dropout, Conv2d, Embedding

class PatchEmbeddings(Module):
    def __init__(self, hyperparameters: dict[str, int]):
        super().__init__()
        
        self.image_size = hyperparameters["image_size"]
        self.patch_size = hyperparameters["patch_size"]
        self.num_channels = hyperparameters["num_channels"]
        self.embedding_dim = hyperparameters["embedding_dim"]

        # Calculate the number of patches
        self.num_patches_height = self.image_size[0] // self.patch_size
        self.num_patches_width = self.image_size[1] // self.patch_size
        self.num_patches = (self.num_patches_height * self.num_patches_width)
        
        # Convolutional layer to map patches to the hidden size dimension
        self.patch_projection = Conv2d(
            in_channels=self.num_channels, 
            out_channels=self.embedding_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
    
    def forward(self, input_image: Tensor) -> Tensor:
        """Forward pass of the PatchEmbeddings module.

        Args:
            input_image (Tensor): Input image tensor of shape (batch_size, num_channels, image_height, image_width).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_patches, embedding_dim).
        """
        
        # Apply convolution to extract patches and project to hidden size dimension
        patch_features = self.patch_projection(input_image)
        
        # Flatten and transpose to convert to (batch_size, num_patches, embedding_dim)
        patch_features = patch_features.flatten(2).transpose(1, 2)
        
        return patch_features


class ImageEmbeddings(Module):
    def __init__(self, hyperparameters: dict[str, int]):
        super().__init__()
        
        self.hyperparameters = hyperparameters
        
        # Initialize patch embeddings based on the given hyperparameters
        self.patch_embeddings = PatchEmbeddings(hyperparameters)
        
        # Learnable position embeddings to capture positional information in the sequence
        self.position_embeddings = Parameter(
            torch.randn(1, self.patch_embeddings.num_patches, hyperparameters["embedding_dim"])
        )
        
        self.dropout = Dropout(hyperparameters["dropout"])

    def forward(self, input_image: Tensor) -> Tensor:
        """Forward pass of the Embeddings module.

        Args:
            input_image (Tensor): Input image tensor of shape (batch_size, num_channels, image_height, image_width).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_patches + 1, embedding_dim).
        """
        
        # Generate patch embeddings from the input image
        patch_embeddings = self.patch_embeddings(input_image)
        
        # Add position embeddings to the sequence
        sequence_with_position = patch_embeddings + self.position_embeddings
        
        # Apply dropout for regularization
        output_embeddings = self.dropout(sequence_with_position)
        
        return output_embeddings


class TargetEmbeddings(Module):
    def __init__(self, hyperparameters: dict[str, int]):
        super().__init__()
        
        self.embeddings = Embedding(hyperparameters['max_length'], hyperparameters['embedding_dim'])
        
        positional_encoding = torch.zeros(hyperparameters['max_length'], hyperparameters['embedding_dim'])
        position = torch.arange(0, hyperparameters['max_length'], dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hyperparameters['embedding_dim'], 2).float() * -(math.log(10000.0) / hyperparameters['embedding_dim']))
        
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))
        
    def forward(self, x: Tensor) -> Tensor:
        return self.embeddings(x) + self.positional_encoding[:, :x.size(1)]
