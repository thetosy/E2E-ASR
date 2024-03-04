import pytorch_lightning as pl
from las import LASEncoderDecoder
from config import DATA_PATH
from data.data_setup import HarperValleyBank
from utils import get_cer_per_sample, run
from torch.utils.data import DataLoader
import torch
import os


class LightningLAS(pl.LightningModule):
    """Train a Listen-Attend-Spell model.
  """

    def __init__(self, n_mels=128, n_fft=256, win_length=256, hop_length=128,
                 wav_max_length=200, transcript_max_length=200,
                 learning_rate=1e-3, batch_size=256, weight_decay=1e-5,
                 encoder_num_layers=2, encoder_hidden_dim=256,
                 encoder_bidirectional=True, encoder_dropout=0,
                 decoder_hidden_dim=256, decoder_num_layers=1,
                 decoder_multi_head=1, decoder_mlp_dim=128,
                 asr_label_smooth=0.1, teacher_force_prob=0.9):
        self.encoder_dropout = encoder_dropout
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_num_layers = decoder_num_layers
        self.decoder_mlp_dim = decoder_mlp_dim
        self.decoder_multi_head = decoder_multi_head
        self.asr_label_smooth = asr_label_smooth
        self.teacher_force_prob = teacher_force_prob

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

        # Instantiate the LAS encoder/decoder.
        self.model = self.create_model()

    def create_model(self):
        model = LASEncoderDecoder(
            self.train_dataset.input_dim,
            self.train_dataset.num_class,
            self.transcript_max_length,
            listener_hidden_dim=self.encoder_hidden_dim,
            listener_bidirectional=self.encoder_bidirectional,
            num_pyramid_layers=self.encoder_num_layers,
            dropout=self.encoder_dropout,
            speller_hidden_dim=self.decoder_hidden_dim,
            speller_num_layers=self.decoder_num_layers,
            mlp_hidden_dim=self.decoder_mlp_dim,
            multi_head=self.decoder_multi_head,
            sos_index=self.train_dataset.sos_index,
            sample_decode=False)
        return model

    def create_datasets(self):
        root = os.path.join(DATA_PATH, 'harper_valley_bank_minified')
        train_dataset = HarperValleyBank(
            root, split='train', n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            append_eos_token=True)  # LAS adds a EOS token to the end of a sequence since it uses attention
        val_dataset = HarperValleyBank(
            root, split='val', n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            append_eos_token=True)
        test_dataset = HarperValleyBank(
            root, split='test', n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            append_eos_token=True)
        return train_dataset, val_dataset, test_dataset

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(),
                                  lr=self.lr, weight_decay=self.weight_decay)
        return [optim], []  # <-- put scheduler in here if you want to use one

    def forward(self, inputs, input_lengths, labels, label_lengths):
        log_probs, embedding = self.model(
            inputs,
            ground_truth=labels,
            teacher_force_prob=self.teacher_force_prob,
        )
        return log_probs, embedding

    def get_loss(self, log_probs, input_lengths, labels, label_lengths):
        loss = self.model.get_loss(log_probs, labels,
                                   self.train_dataset.num_labels,
                                   pad_index=self.train_dataset.pad_index,
                                   label_smooth=self.asr_label_smooth)
        return loss

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
        # outputs is list of metrics from every validation_step (over a
        # validation epoch).
        outputs = self.validation_step_outputs
        metrics = {
            # important that these are torch Tensors!
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
        'learning_rate': 4e-3,  # faster learning rate
        'batch_size': 128,
        'weight_decay': 0,
        'encoder_num_layers': 2,
        'encoder_hidden_dim': 64,
        'encoder_bidirectional': True,
        'encoder_dropout': 0,
        'decoder_hidden_dim': 128,  # must be 2 x encoder_hidden_dim
        'decoder_num_layers': 1,
        'decoder_multi_head': 1,
        'decoder_mlp_dim': 64,
        'asr_label_smooth': 0.1,
        'teacher_force_prob': 0.9,
    }
    run(model=LightningLAS, config=config, epochs=20, ckpt_dir='las', use_gpu=True)
