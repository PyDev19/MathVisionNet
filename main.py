from model import EquationRecognitionModel
from data import train_loader, val_loader, dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.tuner import Tuner

model = EquationRecognitionModel(
    vocab_size=len(dataset.tokenizer.vocab),
    embed_size=256,
    hidden_size=512,
    num_layers=2
)

logger = TensorBoardLogger('logs', name='equation_cnn', log_graph=True)
checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
earlystop_callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')
learningrate_monitor = LearningRateMonitor(logging_interval='step')

trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback, earlystop_callback, learningrate_monitor],
    max_epochs=20,
    enable_checkpointing=True,
    enable_progress_bar=True,
    enable_model_summary=True,
)
tuner = Tuner(trainer)
tuner.lr_find(model, train_loader, val_loader)

trainer.fit(model, train_loader, val_loader)