from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Dropout, GELU, LayerNorm
from model.attention import MultiHeadAttention

class MLP(Module):
    def __init__(self, hyperparameters: dict[str, int]):
        super().__init__()
        
        self.dense_1 = Linear(hyperparameters["embedding_dim"], hyperparameters["dim_feedforward"])
        
        self.activation = GELU()
        
        self.dense_2 = Linear(hyperparameters["dim_feedforward"], hyperparameters["embedding_dim"])
        self.dropout = Dropout(hyperparameters["dropout"])

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
    def __init__(self, hyperparameters: dict[str, int]):
        super().__init__()
        
        self.attention = MultiHeadAttention(hyperparameters)
        self.attention_layer_norm = LayerNorm(hyperparameters["hidden_dim"])
        
        self.mlp = MLP(hyperparameters)
        self.mlp_layer_norm = LayerNorm(hyperparameters["hidden_dim"])
        
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
    def __init__(self, hyperparameters: dict[str, int]):
        super().__init__()
        
        self.encoder_blocks = ModuleList([EncoderBlock(hyperparameters) for _ in range(hyperparameters["num_encoder_layers"])])
    
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
