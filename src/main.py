from datamodule import LaTeXDataModule
from modelmodule import ModelModule

data_dir = "mathwriting-2024-excerpt"
image_size = (128, 512)
model_name = "cnn_lstm"
params_dir = "configs"

data_module = LaTeXDataModule(
    train_dir=f"{data_dir}/train",
    valid_dir=f"{data_dir}/valid",
    test_dir=f"{data_dir}/test",
    vocab_file=f"{data_dir}/vocab.json",
    image_size=image_size,
)

model_module = ModelModule(
    model_name=model_name,
    params_dir=params_dir,
    vocab_size=data_module.get_vocab_size(),
    image_size=image_size,
    sos_token_id=data_module.get_token_id("<SOS>"),
    eos_token_id=data_module.get_token_id("<EOS>"),
)
