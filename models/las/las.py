from utils import reshape_and_apply, label_smooth_loss
from torch import nn
import torch
import torch.functional as F
from torch.distributions import Categorical
from utils import combine_h_and_c


class PyramidLSTMLayer(nn.Module):
    """A Pyramid LSTM layer is a standard LSTM layer that halves the size
  of the input in its hidden embeddings.
  """

    def __init__(self, input_dim, hidden_dim, num_layers=1,
                 bidirectional=True, dropout=0.):
        super().__init__()
        # This is done as part of the design choice for the Pyramid LSTM to reduce the time resolution by half.
        # The doubling of the input_dim is related to the reshaping of the input tensor in the forward method:
        # inputs.contiguous().view(batch_size, maxlen // 2, input_dim * 2). It ensures that the LSTM receives
        # input with twice the original dimensionality.
        self.rnn = nn.LSTM(input_dim * 2, hidden_dim, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=dropout,
                           batch_first=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

    def forward(self, inputs):
        batch_size, maxlen, input_dim = inputs.size()
        inputs = inputs.contiguous().view(batch_size, maxlen // 2, input_dim * 2)
        outputs, hiddens = self.rnn(inputs)
        return outputs, hiddens


class Listener(nn.Module):
    """Stacks 3 layers of PyramidLSTMLayers to reduce resolution 8 times.

  Args:
    input_dim: Number of input features.
    hidden_dim: Number of hidden features.
    num_pyramid_layers: Number of stacked lstm layers. (default: 3)
    dropout: Dropout probability. (default: 0)
  """

    def __init__(
            self, input_dim, hidden_dim, num_pyramid_layers=3, dropout=0.,
            bidirectional=True):
        super().__init__()
        self.rnn_layer0 = PyramidLSTMLayer(input_dim, hidden_dim, num_layers=1,
                                           bidirectional=True, dropout=dropout)

        # dynamically creates and sets attributes for the additional PyramidLSTMLayers,
        # making it flexible for any number of layers specified by num_pyramid_layers
        for i in range(1, num_pyramid_layers):
            setattr(
                self,
                f'rnn_layer{i}',
                PyramidLSTMLayer(hidden_dim * 2, hidden_dim, num_layers=1,
                                 bidirectional=bidirectional, dropout=dropout),
            )

        self.num_pyramid_layers = num_pyramid_layers

    # This design allows the Listener to process input sequences hierarchically,
    # with each PyramidLSTMLayer reducing the temporal resolution by a factor of 2.
    # The bidirectional nature of the PyramidLSTMLayers captures information
    # from both past and future time steps.
    # acoustic model encoder - encodes audio into a denser representation

    def forward(self, inputs):
        outputs, hiddens = self.rnn_layer0(inputs)
        # pass outputs from the previous pLSTM to the next
        for i in range(1, self.num_pyramid_layers):
            outputs, hiddens = getattr(self, f'rnn_layer{i}')(outputs)
        return outputs, hiddens


class AttentionLayer(nn.Module):
    """Attention module that trains an MLP to get attention weights."""

    def __init__(self, input_dim, hidden_dim, multi_head=1):
        super().__init__()

        # transform the input for each head in the multi-head attention.
        self.phi = nn.Linear(input_dim, hidden_dim * multi_head)
        self.psi = nn.Linear(input_dim, hidden_dim)

        if multi_head > 1:
            # reduces the concatenated output of all heads to the original input dimension.
            self.fc_reduce = nn.Linear(input_dim * multi_head, input_dim)

        self.multi_head = multi_head
        self.hidden_dim = hidden_dim

    # listener_feat - encoded audio by 'listener'
    def forward(self, decoder_state, listener_feat):
        input_dim = listener_feat.size(2)
        comp_decoder_state = F.relu(self.phi(decoder_state))
        comp_listener_feat = F.relu(reshape_and_apply(self.psi, listener_feat))

        if self.multi_head == 1:
            # batch matrix multiplication
            energy = torch.bmm(
                comp_decoder_state,
                comp_listener_feat.transpose(1, 2)
            ).squeeze(1)
            attention_score = [F.softmax(energy, dim=-1)]
            # The resulting tensor weights has the same shape as the original listener_feat
            # but with the attention scores repeated along a new dimension,
            # making it suitable for element-wise multiplication with the original features
            weights = attention_score[0].unsqueeze(2).repeat(1, 1, input_dim)
            context = torch.sum(listener_feat * weights, dim=1)
        else:
            attention_score = []
            # split the tensor comp_decoder_state along the last dimension (dim=-1)
            # into chunks, where each chunk has a size self.hidden_dim.
            for att_query in torch.split(comp_decoder_state, self.hidden_dim, dim=-1):
                score = torch.softmax(
                    torch.bmm(att_query,
                              comp_listener_feat.transpose(1, 2)).squeeze(dim=1),
                )
                attention_score.append(score)
            # calculate attention score for each head
            projected_src = []
            for att_s in attention_score:
                weights = att_s.unsqueeze(2).repeat(1, 1, input_dim)
                proj = torch.sum(listener_feat * weights, dim=1)
                projected_src.append(proj)
            # combine all attention head score
            context = self.fc_reduce(torch.cat(projected_src, dim=-1))

        # context is the entries of listener input weighted by attention
        return attention_score, context


class Speller(nn.Module):
    """Decoder that uses a LSTM with attention to convert a sequence of
  hidden embeddings to a sequence of probabilities for output classes.
  """

    def __init__(
            self, num_labels, label_maxlen, speller_hidden_dim,
            listener_hidden_dim, mlp_hidden_dim, num_layers=1, multi_head=1,
            sos_index=0, sample_decode=False):
        super().__init__()
        self.rnn = nn.LSTM(num_labels + speller_hidden_dim,
                           speller_hidden_dim, num_layers=num_layers,
                           batch_first=True)
        self.attention = AttentionLayer(listener_hidden_dim * 2, mlp_hidden_dim,
                                        multi_head=multi_head)
        self.fc_out = nn.Linear(speller_hidden_dim * 2, num_labels)
        self.num_labels = num_labels
        self.label_maxlen = label_maxlen
        self.sample_decode = sample_decode
        self.sos_index = sos_index

    def step(self, inputs, last_hiddens, listener_feats):
        outputs, cur_hiddens = self.rnn(inputs, last_hiddens)
        attention_score, context = self.attention(outputs, listener_feats)
        features = torch.cat((outputs.squeeze(1), context), dim=-1)
        logits = self.fc_out(features)
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs, cur_hiddens, context, attention_score

    def forward(
            self, listener_feats, ground_truth=None, teacher_force_prob=0.9):
        device = listener_feats.device
        if ground_truth is None:
            teacher_force_prob = 0
        teacher_force = np.random.random_sample() < teacher_force_prob

        batch_size = listener_feats.size(0)
        with torch.no_grad():
            output_toks = torch.zeros((batch_size, 1, self.num_labels), device=device)
            # set prob of the beginning of the sentence index token to 1
            output_toks[:, 0, self.sos_index] = 1
            # Concatenates the output_toks tensor with the first time step of the
            # listener_feats tensor along the last dimension (dim=-1)
            # listener_feats[:, 0:1, :] = (batch_size, 1, feature_dim)
            # rnn_inputs = (batch_size, 1, self.num_labels + feature_dim)

        rnn_inputs = torch.cat([output_toks, listener_feats[:, 0:1, :]], dim=-1)

        hidden_state = None
        log_probs_seq = []

        if (ground_truth is None) or (not teacher_force_prob):
            max_step = int(self.label_maxlen)
        else:
            max_step = int(ground_truth.size(1))

        for step in range(max_step):
            log_probs, hidden_state, context, _ = self.step(
                rnn_inputs, hidden_state, listener_feats)
            log_probs_seq.append(log_probs.unsqueeze(1))

            if teacher_force:
                gt_tok = ground_truth[:, step:step + 1].float()
                output_tok = torch.zeros_like(log_probs)
                for idx, i in enumerate(gt_tok):
                    output_tok[idx, int(i.item())] = 1
                output_tok = output_tok.unsqueeze(1)
            else:
                if self.sample_decode:
                    probs = torch.exp(log_probs)
                    sampled_tok = Categorical(probs).sample()
                else:  # Pick max probability (greedy decoding)
                    sampled_tok = log_probs.topk(1)[1]

                output_tok = torch.zeros_like(log_probs)
                for idx, i in enumerate(sampled_tok):
                    output_tok[idx, int(i.item())] = 1
                output_tok = output_tok.unsqueeze(1)

            rnn_inputs = torch.cat([output_tok, context.unsqueeze(1)], dim=-1)

        # batch_size x maxlen x num_labels
        log_probs_seq = torch.cat(log_probs_seq, dim=1)

        return log_probs_seq.contiguous()


class LASEncoderDecoder(nn.Module):
    def __init__(
            self, input_dim, num_class, label_maxlen, listener_hidden_dim=128,
            listener_bidirectional=True, num_pyramid_layers=3, dropout=0,
            speller_hidden_dim=256, speller_num_layers=1, mlp_hidden_dim=128,
            multi_head=1, sos_index=0, sample_decode=False):
        super().__init__()
        # Encoder.
        self.listener = Listener(input_dim, listener_hidden_dim,
                                 num_pyramid_layers=num_pyramid_layers,
                                 dropout=dropout,
                                 bidirectional=listener_bidirectional)
        # Decoder.
        self.speller = Speller(num_class, label_maxlen, speller_hidden_dim,
                               listener_hidden_dim, mlp_hidden_dim,
                               num_layers=speller_num_layers,
                               multi_head=multi_head,
                               sos_index=sos_index,
                               sample_decode=sample_decode)
        self.embedding_dim = listener_hidden_dim * 4

    def forward(self, inputs, ground_truth=None, teacher_force_prob=0.9):
        outputs, (h, c) = self.listener(inputs)
        log_probs = self.speller(outputs, ground_truth=ground_truth, teacher_force_prob=teacher_force_prob)

        combined_h_and_c = combine_h_and_c(h, c)
        return log_probs, combined_h_and_c

    def get_loss(
            self, log_probs, labels, num_labels, pad_index=0, label_smooth=0.1):
        batch_size = log_probs.size(0)
        labels_maxlen = labels.size(1)

        if label_smooth == 0.0:
            loss = F.nll_loss(log_probs.view(batch_size * labels_maxlen, -1),
                              labels.long().view(batch_size * labels_maxlen),
                              ignore_index=pad_index)
        else:
            # label_smooth_loss is the sample as F.nll_loss but with a temperature
            # parameter that makes the log probability distribution "sharper".
            loss = label_smooth_loss(log_probs, labels.float(), num_labels,
                                     smooth_param=label_smooth)
        return loss

    def decode(self, log_probs, input_lengths, labels, label_lengths,
               sos_index, eos_index, pad_index, eps_index):
        # Use greedy decoding.
        decoded = torch.argmax(log_probs, dim=2)
        batch_size = decoded.size(0)
        # Collapse each decoded sequence using CTC rules.
        hypotheses = []
        hypothesis_lengths = []
        references = []
        reference_lengths = []
        for i in range(batch_size):
            decoded_i = decoded[i]
            hypothesis_i = []
            for tok in decoded_i:
                if tok.item() == sos_index:
                    continue
                if tok.item() == pad_index:
                    continue
                if tok.item() == eos_index:
                    # once we reach an EOS token, we are done generating.
                    break
                hypothesis_i.append(tok.item())
            hypotheses.append(hypothesis_i)
            hypothesis_lengths.append(len(hypothesis_i))

            if labels is not None:
                label_i = labels[i]
                reference_i = [tok.item() for tok in labels[i]
                               if tok.item() != sos_index and
                               tok.item() != eos_index and
                               tok.item() != pad_index]
                references.append(reference_i)
                reference_lengths.append(len(reference_i))

        if labels is None:  # Run at inference time.
            references, reference_lengths = None, None

        return hypotheses, hypothesis_lengths, references, reference_lengths
