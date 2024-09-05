from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Dropout, GELU, LayerNorm
from image2latex.attention import MultiHeadAttention

class MLP(Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        
        self.dense_1 = Linear(embedding_dim, hidden_dim)
        
        self.activation = GELU()
        
        self.dense_2 = Linear(hidden_dim, embedding_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MLP module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        x = self.dense_1(x)
        
        x = self.activation(x)
        
        x = self.dense_2(x)
        x = self.dropout(x)
        
        return x

class EncoderBlock(Module):
    def __init__(self, embedding_dim: int, num_heads: int, qkv_bias: bool, dropout: float, hidden_dim: int):
        super().__init__()
        
        self.attention = MultiHeadAttention(embedding_dim, num_heads, qkv_bias, dropout)
        self.attention_layer_norm = LayerNorm(hidden_dim)
        
        self.mlp = MLP(embedding_dim, hidden_dim, dropout)
        self.mlp_layer_norm = LayerNorm(hidden_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the EncoderBlock module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        # Apply self-attention mechanism
        attention_output = self.attention(x)
        x = self.attention_layer_norm(x + attention_output)
        
        # Apply MLP
        mlp_output = self.mlp(x)
        x = self.mlp_layer_norm(x + mlp_output)
        
        return x

class EncoderLayer(Module):
    def __init__(self, embedding_dim: int, num_heads: int, qkv_bias: bool, dropout: float, hidden_dim: int, num_encoder_layers: int):
        super().__init__()
        
        self.encoder_blocks = ModuleList([
            EncoderBlock(embedding_dim, num_heads, qkv_bias, dropout, hidden_dim) for _ in range(num_encoder_layers)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the EncoderLayer module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        
        return x
