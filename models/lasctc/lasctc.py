import torch.functional as F
import torch.nn as nn
from utils import get_ctc_loss
from models.las.las import LASEncoderDecoder


class CTCDecoder(nn.Module):
    """
  This is a small decoder (just one linear layer) that takes
  the listener embedding from LAS and imposes a CTC
  objective on the decoding.

  NOTE: This is only to be used for the Joint CTC-Attention model.
  """

    def __init__(self, listener_hidden_dim, num_class, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(listener_hidden_dim, num_class)
        self.dropout = nn.Dropout(dropout)
        self.listener_hidden_dim = listener_hidden_dim
        self.num_class = num_class

    def forward(self, listener_outputs):
        batch_size, maxlen, _ = listener_outputs.size()
        logits = self.fc(self.dropout(listener_outputs))
        logits = logits.view(batch_size, maxlen, self.num_class)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    def get_loss(
            self, log_probs, input_lengths, labels, label_lengths, blank=0):
        return get_ctc_loss(
            log_probs, labels, input_lengths, label_lengths, blank)


class JointCTCAttention(LASEncoderDecoder):
    """Joint CTC and LAS model that optimizes the LAS objective but
  regularized by the conditional independence of a CTC decoder. One
  can interpret CTC as regularizer on LAS.
  """

    def __init__(
            self, input_dim, num_class, label_maxlen, listener_hidden_dim=128,
            listener_bidirectional=True, num_pyramid_layers=3, dropout=0,
            speller_hidden_dim=256, speller_num_layers=1, mlp_hidden_dim=128,
            multi_head=1, sos_index=0, sample_decode=False):
        super().__init__(
            input_dim,
            num_class,
            label_maxlen,
            listener_hidden_dim=listener_hidden_dim,
            listener_bidirectional=listener_bidirectional,
            num_pyramid_layers=num_pyramid_layers,
            dropout=dropout,
            speller_hidden_dim=speller_hidden_dim,
            speller_num_layers=speller_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            multi_head=multi_head,
            sos_index=sos_index,
            sample_decode=sample_decode,
        )
        self.ctc_decoder = CTCDecoder(listener_hidden_dim * 2, num_class)
        self.num_pyramid_layers = num_pyramid_layers
        self.embedding_dim = listener_hidden_dim * 4

    def forward(self, inputs, ground_truth=None, teacher_force_prob=0.9, ):
        outputs, (h, c) = self.listener(inputs)
        ctc_log_probs = self.ctc_decoder(outputs)
        las_log_probs = self.speller(outputs)
        listener_hc = self.combine_h_and_c(h, c)
        return ctc_log_probs, las_log_probs, listener_hc

    def get_loss(self, ctc_log_probs, las_log_probs, input_lengths, labels, label_lengths,
            num_labels, pad_index=0, blank_index=0, label_smooth=0.1):
        ctc_loss = self.ctc_decoder.get_loss(
            ctc_log_probs,
            # pyramid encode cuts timesteps in 1/2 each way
            input_lengths // (2 ** self.num_pyramid_layers),
            labels,
            label_lengths,
            blank=blank_index,
        )
        las_loss = super().get_loss(las_log_probs, labels, num_labels,
                                    pad_index=pad_index, label_smooth=label_smooth)

        return ctc_loss, las_loss

    def decode(self, log_probs, input_lengths, labels, label_lengths,
               sos_index, eos_index, pad_index, eps_index):
        las_log_probs = log_probs[1]
        return super().decode(las_log_probs, input_lengths, labels, label_lengths,
                              sos_index, eos_index, pad_index, eps_index)
