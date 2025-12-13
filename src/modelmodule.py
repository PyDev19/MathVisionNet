import yaml
import os
from models.cnn_lstm import CNNLSTM_EncoderDecoder


class ModelModule:
    def __init__(
        self,
        model_name: str,
        params_dir: str,
        vocab_size: int,
        image_size: tuple[int, int],
    ):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.image_size = image_size

        params_path = os.path.join(params_dir, f"{self.model_name}.yaml")
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.model = self._initialize_model()

    def _initialize_model(self):
        match self.model_name:
            case "cnn_lstm":
                return CNNLSTM_EncoderDecoder(
                    **self.params["model"],
                    vocab_size=self.vocab_size,
                    image_size=self.image_size,
                )

            case _:
                raise ValueError(f"Model '{self.model_name}' is not supported.")
