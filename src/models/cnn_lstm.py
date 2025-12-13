import torch
from torch.nn import Module, Conv2d, LSTM, Linear, SiLU, MaxPool2d, ModuleList, Embedding


class CNNBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_size: int,
    ):
        super(CNNBlock, self).__init__()

        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = SiLU()
        self.pool = MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)

        return x


class CNNLSTM_EncoderDecoder(Module):
    def __init__(
        self,
        input_sizes: list[int],
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
        image_size: tuple[int, int],
        sos_token_id: int,
        eos_token_id: int
    ):
        super(CNNLSTM_EncoderDecoder, self).__init__()

        self.cnn_encoder = ModuleList(
            [
                CNNBlock(
                    in_channels=input_sizes[i],
                    out_channels=input_sizes[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pool_size=2,
                )
                for i in range(len(input_sizes) - 1)
            ]
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, input_sizes[0], *image_size)
            
            for cnn in self.cnn_encoder:
                dummy = cnn(dummy)
            
            self.flatten_size = dummy.numel()

        self.encoder_projection = Linear(self.flatten_size, hidden_size)
        self.embedding = Embedding(vocab_size, hidden_size)
        
        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.output_layer = Linear(hidden_size, vocab_size)
        
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        
    def forward(self, x, target_seq=None, max_seq_len=1000):
        for cnn in self.cnn_encoder:
            x = cnn(x)

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        encoded = self.encoder_projection(x)
        h_0 = encoded.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        
        if target_seq is not None:
            embeddings = self.embedding(target_seq)
            lstm_out, _ = self.lstm(embeddings, (h_0, c_0))
            output = self.output_layer(lstm_out)
        else:
            outputs = []
            input_token = torch.zeros(batch_size, self.sos_token_id, dtype=torch.long, device=x.device)
            
            for _ in range(max_seq_len):
                embedding = self.embedding(input_token)
                lstm_out, (h_0, c_0) = self.lstm(embedding, (h_0, c_0))
                
                output_token = self.output_layer(lstm_out)
                
                outputs.append(output_token)
                input_token = output_token.argmax(dim=-1)
                
                if (torch.eq(input_token, self.eos_token_id).all()):
                    break
            
            output = torch.cat(outputs, dim=1)
        
        return output
