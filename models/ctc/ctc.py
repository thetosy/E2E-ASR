import torch.nn as nn
import torch
import torch.nn.functional as F
import utils


class CTCEncoderDecoder(nn.Module):
    """
  Encoder-Decoder model trained with CTC objective.

  Args:
    input_dim: integer
                number of input features
    num_class: integer
                size of transcription vocabulary
    num_layers: integer (default: 2)
                number of layers in encoder LSTM
    hidden_dim: integer (default: 128)
                number of hidden dimensions for encoder LSTM
    bidirectional: boolean (default: True)
                    is the encoder LSTM bidirectional?
  """

    def __init__(
            self, input_dim, num_class, num_layers=2, hidden_dim=128,
            bidirectional=True):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               bidirectional=bidirectional, batch_first=True)
        self.decoder = nn.Linear(in_features=hidden_dim * 2, out_features=num_class)
        self.dropout = nn.Dropout(p=0.5)
        self.input_dim = input_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = hidden_dim * num_layers * 2 * (2 if bidirectional else 1)

    def forward(self, inputs, input_lengths):
        batch_size, max_length, _ = inputs.size()
        inputs = torch.nn.utils.rnn.pack_padded_sequence(
            inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.encoder(inputs)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0.0,
                                                            total_length=max_length)
        outputs = self.dropout(outputs)
        # logits = (batch_size, max_length, num_class)
        logits = self.decoder(outputs)
        # get probablility for each char
        log_probs = F.log_softmax(logits)
        embedding = utils.combine_h_and_c(h, c)
        return log_probs, embedding

    def get_loss(
            self, log_probs, targets, input_lengths, target_lengths, blank=0):
        return utils.get_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank)

    def decode(self, log_probs, input_lengths, labels, label_lengths,
               sos_index, eos_index, pad_index, eps_index):
        # Use greedy decoding.
        # find the indices of the maximum values along a specific dimension of a tensor
        # which here is num_class
        # we get (batch_size, sequence_length)
        decoded = torch.argmax(log_probs, dim=2)
        batch_size = decoded.size(0)
        # Collapse each decoded sequence using CTC rules.
        hypotheses = []
        for i in range(batch_size):
            hypotheses_i = self.ctc_collapse(decoded[i], input_lengths[i].item(),
                                             blank_index=eps_index)
            hypotheses.append(hypotheses_i)

        hypothesis_lengths = input_lengths.cpu().numpy().tolist()
        if labels is None:  # Run at inference time.
            references, reference_lengths = None, None
        else:
            references = labels.cpu().numpy().tolist()
            reference_lengths = label_lengths.cpu().numpy().tolist()

        return hypotheses, hypothesis_lengths, references, reference_lengths

    def ctc_collapse(self, seq, seq_len, blank_index=0):
        result = []
        for i, tok in enumerate(seq[:seq_len]):
            if tok.item() != blank_index:  # remove blanks
                if i != 0 and tok.item() == seq[i - 1].item():  # remove dups
                    pass
                else:
                    result.append(tok.item())
        return result
