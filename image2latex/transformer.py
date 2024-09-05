import torch
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, Linear, Module
from image2latex.embedding import ImageEmbeddings, TargetEmbeddings
from image2latex.encoder import EncoderLayer

class VisionTransformerDecoder(Module):
    def __init__(self, hyperparameters: dict[str, int]):
        super().__init__()
        
        self.model_hyperparameters = hyperparameters
        
        self.image_embeddings = ImageEmbeddings(
            hyperparameters["image_size"],
            hyperparameters["patch_size"],
            hyperparameters["num_channels"],
            hyperparameters["embedding_dim"],
            hyperparameters["dropout"]
        )
        self.target_embeddings = TargetEmbeddings(
            hyperparameters["max_length"],
            hyperparameters["embedding_dim"],
        )
        
        self.encoder = EncoderLayer(
            hyperparameters["embedding_dim"],
            hyperparameters["num_heads"],
            hyperparameters["qkv_bias"],
            hyperparameters["dropout"],
            hyperparameters["hidden_dim"],
            hyperparameters["num_encoder_layers"],
        )
        
        self.decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=hyperparameters["embedding_dim"],
                nhead=hyperparameters["num_heads"],
                dropout=hyperparameters["dropout"],
                dim_feedforward=hyperparameters["dim_feedforward"],
            ),
            num_layers=hyperparameters["num_decoder_layers"],
        )
        
        self.output_projection = Linear(hyperparameters["embedding_dim"], hyperparameters["vocab_size"])
    
    def generate_square_subsequent_mask(self, sequences: Tensor) -> Tensor:
        mask = torch.triu(torch.ones(sequences.size(0), sequences.size(0)) * float('-inf'), diagonal=1)
        
        return mask.to(sequences.device)

    def create_padding_mask(self, sequences: Tensor) -> Tensor:
        return (sequences == 0).to(sequences.device)
    
    def forward(self, input_image: Tensor, target_sequence: Tensor) -> Tensor:
        image_embeddings = self.image_embeddings(input_image)
        target_embeddings = self.target_embeddings(target_sequence)
        
        encoder_output = self.encoder(image_embeddings)
        
        target_embeddings = target_embeddings.transpose(0, 1) # (sequence_length, batch_size, embedding_dim)
        decoder_image = encoder_output.transpose(0, 1) # (sequence_length, batch_size, embedding_dim)
        
        target_mask = self.generate_square_subsequent_mask(target_embeddings)
        target_padding_mask = self.create_padding_mask(target_sequence)
                
        decoder_output = self.decoder(
            target_embeddings,
            decoder_image,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
        )
        
        output_logits = self.output_projection(decoder_output)
        
        output_logits = output_logits.transpose(0, 1) # (batch_size, sequence_length, vocab_size)

        return output_logits
