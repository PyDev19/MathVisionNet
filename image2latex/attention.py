import math
import torch
from torch import Tensor
from torch.nn import Module, Linear, Dropout
from torch.nn.functional import softmax

class MultiHeadAttention(Module):
    def __init__(self, embedding_dim: int, num_heads: int, qkv_bias: bool, dropout: float):
        super().__init__()
        self.num_heads = num_heads

        # Compute the size of each attention head
        self.size_per_head = embedding_dim // self.num_heads
        self.attention_size = self.num_heads * self.size_per_head

        # Linear layer to project input into query, key, and value vectors
        self.qkv_projection = Linear(embedding_dim, self.attention_size * 3, bias=qkv_bias)
        self.attn_dropout = Dropout(dropout)

        # Linear layer to project the concatenated output of all attention heads
        self.output_projection = Linear(self.attention_size, embedding_dim)
        self.output_dropout = Dropout(dropout)

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the MultiHeadAttention module.

        Args:
            input_tensor (Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            tuple[Tensor, Tensor]: Output tensor of shape (batch_size, sequence_length, hidden_size) and attention probabilities of shape (batch_size, num_attention_heads, sequence_length, sequence_length).
        """
        # Project input_tensor to query, key, value tensors
        qkv_combined = self.qkv_projection(input_tensor)
        
        # Split qkv_combined into query, key, and value tensors
        query_tensor, key_tensor, value_tensor = torch.chunk(qkv_combined, 3, dim=-1)
        batch_size, sequence_length, _ = query_tensor.size()
        
        # Reshape for multi-head attention and transpose to bring head dimension first
        query_tensor = query_tensor.view(batch_size, sequence_length, self.num_heads, self.size_per_head).transpose(1, 2)
        key_tensor = key_tensor.view(batch_size, sequence_length, self.num_heads, self.size_per_head).transpose(1, 2)
        value_tensor = value_tensor.view(batch_size, sequence_length, self.num_heads, self.size_per_head).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_tensor, key_tensor.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.size_per_head)
        
        # Normalize the attention scores to probabilities
        attention_probabilities = softmax(attention_scores, dim=-1)
        attention_probabilities = self.attn_dropout(attention_probabilities)

        # Compute the attention output
        attention_output = torch.matmul(attention_probabilities, value_tensor)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.attention_size)
        
        # Apply final linear projection and dropout
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        return attention_output
