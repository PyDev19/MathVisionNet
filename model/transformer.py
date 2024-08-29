import torch
from torch import Tensor
from torch.nn import Module, TransformerDecoder, TransformerDecoderLayer, Linear, Transformer
from pytorch_lightning import LightningModule
from model.embedding import ImageEmbeddings, TargetEmbeddings
from model.encoder import EncoderLayer

class Image2LaTeXVisionTransformer(LightningModule):
    def __init__(self, hyperparameters: dict[str, int]):
        super().__init__()
        
        self.image_embeddings = ImageEmbeddings(hyperparameters)
        self.target_embeddings = TargetEmbeddings(hyperparameters)
        
        self.encoder = EncoderLayer(hyperparameters)
        
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
        
        self.example_input_array = (torch.randn(1, 3, 128, 512), torch.randint(1, 91, (1, 100)))
    
    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, input_image: Tensor, target_sequence: Tensor) -> Tensor:
        image_embeddings = self.image_embeddings(input_image)
        target_embeddings = self.target_embeddings(target_sequence)
        
        encoder_output = self.encoder(image_embeddings)
        
        target_embeddings = target_embeddings.transpose(0, 1)
        decoder_image = encoder_output[0].transpose(0, 1)
        
        target_mask = self.generate_square_subsequent_mask(target_sequence.size(1))
        
        decoder_output = self.decoder(
            target_embeddings,
            decoder_image,
            tgt_mask=target_mask,
            tgt_is_causal=True,
        )
        
        output_logits = self.output_projection(decoder_output)

        return output_logits