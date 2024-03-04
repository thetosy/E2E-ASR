import pytorch_lightning as pl
from ctc import CTCEncoderDecoder
from config import DATA_PATH
from data.data_setup import HarperValleyBank
from utils import get_cer_per_sample, run
from torch.utils.data import DataLoader
import torch
import os


class LightningCTC(pl.LightningModule):
    """PyTorch Lightning class for training a CTC model.

  Args:
    n_mels: number of mel frequencies. (default: 128)
    n_fft: number of fourier features. (default: 256)
    win_length: number of frames in a window. (default: 256)
    hop_length: number of frames to hop in computing spectrogram. (default: 128)
    wav_max_length: max number of timesteps in a waveform spectrogram. (default: 200)
    transcript_max_length: max number of characters in decoded transcription. (default: 200)
    learning_rate: learning rate for Adam optimizer. (default: 1e-3)
    batch_size: batch size used in optimization and evaluation. (default: 256)
    weight_decay: weight decay for Adam optimizer. (default: 1e-5)
    encoder_num_layers: number of layers in LSTM encoder. (default: 2)
    encoder_hidden_dim: number of hidden dimensions in LSTM encoder. (default: 256)
    encoder_bidirectional: directionality of LSTM encoder. (default: True)
  """

    def __init__(self, n_mels=128, n_fft=256, win_length=256, hop_length=128,
                 wav_max_length=200, transcript_max_length=200,
                 learning_rate=1e-3, batch_size=256, weight_decay=1e-5,
                 encoder_num_layers=2, encoder_hidden_dim=256,
                 encoder_bidirectional=True):
        super().__init__()
        self.save_hyperparameters()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.lr = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.wav_max_length = wav_max_length
        self.transcript_max_length = transcript_max_length
        self.train_dataset, self.val_dataset, self.test_dataset = \
            self.create_datasets()
        self.encoder_num_layers = encoder_num_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_bidirectional = encoder_bidirectional
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Instantiate the CTC encoder/decoder.
        self.model = self.create_model()

    def create_model(self):
        model = CTCEncoderDecoder(
            self.train_dataset.input_dim,
            self.train_dataset.num_class,
            num_layers=self.encoder_num_layers,
            hidden_dim=self.encoder_hidden_dim,
            bidirectional=self.encoder_bidirectional)
        return model

    def create_datasets(self):
        root = os.path.join(DATA_PATH, 'harper_valley_bank_minified')
        train_dataset = HarperValleyBank(
            root, split='train', n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            append_eos_token=False)
        val_dataset = HarperValleyBank(
            root, split='val', n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            append_eos_token=False)
        test_dataset = HarperValleyBank(
            root, split='test', n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            append_eos_token=False)
        return train_dataset, val_dataset, test_dataset

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(),
                                  lr=self.lr, weight_decay=self.weight_decay)
        return [optim], []  # <-- put scheduler in here if you want to use one

    def get_loss(self, log_probs, input_lengths, labels, label_lengths):
        loss = self.model.get_loss(log_probs, labels, input_lengths, label_lengths,
                                   blank=self.train_dataset.eps_index)
        return loss

    def forward(self, inputs, input_lengths, labels, label_lengths):
        log_probs, embedding = self.model(inputs, input_lengths)
        return log_probs, embedding

    def get_primary_task_loss(self, batch, split='train'):
        """Returns ASR model losses, metrics, and embeddings for a batch."""
        inputs, input_lengths = batch[0], batch[1]
        labels, label_lengths = batch[2], batch[3]

        if split == 'train':
            log_probs, embedding = self.forward(
                inputs, input_lengths, labels, label_lengths)
        else:
            # do not pass labels to not teacher force after training
            log_probs, embedding = self.forward(
                inputs, input_lengths, None, None)

        loss = self.get_loss(log_probs, input_lengths, labels, label_lengths)

        # Compute CER (no gradient necessary).
        with torch.no_grad():
            hypotheses, hypothesis_lengths, references, reference_lengths = \
                self.model.decode(
                    log_probs, input_lengths, labels, label_lengths,
                    self.train_dataset.sos_index,
                    self.train_dataset.eos_index,
                    self.train_dataset.pad_index,
                    self.train_dataset.eps_index)
            cer_per_sample = get_cer_per_sample(
                hypotheses, hypothesis_lengths, references, reference_lengths)
            cer = cer_per_sample.mean()
            metrics = {f'{split}_loss': loss, f'{split}_cer': cer}

        return loss, metrics, embedding

    # Overwrite TRAIN
    def training_step(self, batch, batch_idx):
        loss, metrics, _ = self.get_primary_task_loss(batch, split='train')
        self.log_dict(metrics)
        return loss

    # Overwrite VALIDATION: get next minibatch
    def validation_step(self, batch, batch_idx):
        loss, metrics, _ = self.get_primary_task_loss(batch, split='val')
        self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        _, metrics, _ = self.get_primary_task_loss(batch, split='test')
        self.test_step_outputs.append(metrics)
        return metrics

    # Overwrite: e.g. accumulate stats (avg over CER and loss)
    def on_validation_epoch_end(self):
        """Called at the end of validation step to aggregate outputs."""
        outputs = self.validation_step_outputs
        metrics = {
            'val_loss': torch.tensor([elem['val_loss']
                                      for elem in outputs]).float().mean(),
            'val_cer': torch.tensor([elem['val_cer']
                                     for elem in outputs]).float().mean()
        }
        self.log_dict(metrics)

    # Overwrite: e.g. accumulate stats (avg over CER and loss)
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        metrics = {
            'test_loss': torch.tensor([elem['test_loss']
                                       for elem in outputs]).float().mean(),
            'test_cer': torch.tensor([elem['test_cer']
                                      for elem in outputs]).float().mean()
        }
        self.log_dict(metrics)

    # dataloader hooks
    def train_dataloader(self):
        # - important to shuffle to not overfit!
        # - drop the last batch to preserve consistent batch sizes
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                            shuffle=True, pin_memory=True, drop_last=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                            shuffle=False, pin_memory=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                            shuffle=False, pin_memory=True)
        return loader


if __name__ == "__main__":
    config = {
        'n_mels': 128,
        'n_fft': 256,
        'win_length': 256,
        'hop_length': 128,
        'wav_max_length': 512,
        'transcript_max_length': 200,
        'learning_rate': 1e-3,
        'batch_size': 128,
        'weight_decay': 0,
        'encoder_num_layers': 2,
        'encoder_hidden_dim': 256,
        'encoder_bidirectional': True,
    }

    run(model=LightningCTC, config=config, ckpt_dir='ctc', epochs=20, use_gpu=True)
