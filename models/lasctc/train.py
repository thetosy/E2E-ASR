from models.las.train import LightningLAS
from lasctc import JointCTCAttention
from utils import run


class LightningCTCLAS(LightningLAS):

    def __init__(self, n_mels=128, n_fft=256, win_length=256, hop_length=128,
                 wav_max_length=200, transcript_max_length=200,
                 learning_rate=1e-3, batch_size=256, weight_decay=1e-5,
                 encoder_num_layers=2, encoder_hidden_dim=256,
                 encoder_bidirectional=True, encoder_dropout=0,
                 decoder_hidden_dim=256, decoder_num_layers=1,
                 decoder_multi_head=1, decoder_mlp_dim=128,
                 asr_label_smooth=0.1, teacher_force_prob=0.9,
                 ctc_weight=0.5):
        super().__init__(
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            wav_max_length=wav_max_length,
            transcript_max_length=transcript_max_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            encoder_num_layers=encoder_num_layers,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_bidirectional=encoder_bidirectional,
            encoder_dropout=encoder_dropout,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_num_layers=decoder_num_layers,
            decoder_multi_head=decoder_multi_head,
            decoder_mlp_dim=decoder_mlp_dim,
            asr_label_smooth=asr_label_smooth,
            teacher_force_prob=teacher_force_prob)
        self.save_hyperparameters()
        self.ctc_weight = ctc_weight

    def create_model(self):
        model = JointCTCAttention(
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

    def forward(self, inputs, input_lengths, labels, label_lengths):
        ctc_log_probs, las_log_probs, embedding = self.model(
            inputs,
            ground_truth=labels,
            teacher_force_prob=self.teacher_force_prob)
        return (ctc_log_probs, las_log_probs), embedding

    def get_loss(self, log_probs, input_lengths, labels, label_lengths):
        (ctc_log_probs, las_log_probs) = log_probs
        ctc_loss, las_loss = self.model.get_loss(
            ctc_log_probs,
            las_log_probs,
            input_lengths,
            labels,
            label_lengths,
            self.train_dataset.num_labels,
            pad_index=self.train_dataset.pad_index,
            blank_index=self.train_dataset.eps_index,
            label_smooth=self.asr_label_smooth)
        # interpolation of loss
        loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * las_loss
        return loss


if __name__ == "__main__":
    config = {
        'n_mels': 128,
        'n_fft': 256,
        'win_length': 256,
        'hop_length': 128,
        'wav_max_length': 512,
        'transcript_max_length': 200,
        'learning_rate': 4e-3,
        'batch_size': 128,
        'weight_decay': 0,
        'encoder_num_layers': 2,  # can't shrink output too much...
        'encoder_hidden_dim': 64,
        'encoder_bidirectional': True,
        'encoder_dropout': 0,
        'decoder_hidden_dim': 128,  # must be 2 x encoder_hidden_dim
        'decoder_num_layers': 1,
        'decoder_multi_head': 1,
        'decoder_mlp_dim': 64,
        'asr_label_smooth': 0.1,
        'teacher_force_prob': 0.9,
        # you may wish to play with these weights; try to keep the sum
        # of them equal to one.
        'ctc_weight': 0.5,  # equal weight between ctc and las?
    }
    run(model=LightningCTCLAS, config=config, epochs=20, ckpt_dir='ctc_las', use_gpu=True)

