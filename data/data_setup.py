from config import DATA_PATH, MODEL_PATH
from utils import prune_transcripts, pad_wav, pad_transcript_label, get_transcript_labels, get_cer_per_sample
import h5py
import os
import numpy as np
from torch.utils.data import Dataset
import librosa

# HarperValleyBank character vocabulary
VOCAB = [' ', "'", '~', '-', '.', '<', '>', '[', ']', 'a', 'b', 'c', 'd', 'e',
         'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
         't', 'u', 'v', 'w', 'x', 'y', 'z']

SILENT_VOCAB = ['[baby]', '[ringing]', '[laughter]', '[kids]', '[music]',
                '[noise]', '[unintelligible]', '[dogs]', '[cough]']


def indices_to_chars(indices):
    full_vocab = ['<eps>', '<sos>', '<eos>', '<pad>'] + VOCAB + SILENT_VOCAB
    chars = [full_vocab[ind] for ind in indices]
    return chars


class HarperValleyBank(Dataset):
    """Dataset to be used to train CTC, LAS, and MTL.

  Args:
    root: string
          path to the data files.
    split: string (default: train)
            choices: train | val | test
            which split of data to load
    n_mels: integer (default: 128)
            number of mel frequencies
    n_fft: integer (default: 256)
            number of fourier components
    win_length: integer (default: 256)
                should be <= n_fft
    hop_length: integer (default: 128)
                number of frames to skip in between
    wav_max_length: integer (default: 200)
                    maximum number of timesteps in a waveform
    transcript_max_length: integer (default: 200)
                            maximum number of timesteps in a transcript
    append_eos_token: boolean (default: False)
                      add EOS token to the end of every transcription
                      this is used for LAS (and LAS+CTC models)
  """

    def __init__(
            self, root, split='train', n_mels=128, n_fft=256, win_length=256,
            hop_length=128, wav_max_length=200, transcript_max_length=200,
            append_eos_token=False):
        super().__init__()
        print(f'> Constructing HarperValleyBank {split} dataset...')

        self.label_data = np.load(os.path.join(root, 'labels.npz'))
        self.root = root
        self.wav_max_length = wav_max_length
        self.transcript_max_length = transcript_max_length

        self.input_dim = n_mels
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # Prune away very short examples.
        # This returns a list of indices of examples longer than 3 words.
        valid_indices = prune_transcripts(self.label_data['human_transcripts'])

        # Decides which indices belong to which split.
        train_indices, val_indices, test_indices = self.split_data(valid_indices)

        if split == 'train':
            indices = train_indices
        elif split == 'val':
            indices = val_indices
        elif split == 'test':
            indices = test_indices
        else:
            raise Exception(f'Split {split} not supported.')

        raw_human_transcripts = self.label_data['human_transcripts'].tolist()
        human_transcript_labels = get_transcript_labels(
            raw_human_transcripts, VOCAB, SILENT_VOCAB)

        # Increment all indices by 4 to reserve the following special tokens:
        #   0 for epsilon
        #   1 for start-of-sentence (SOS)
        #   2 for end-of-sentence (EOS)
        #   3 for padding
        num_special_tokens = 4
        human_transcript_labels = [list(np.array(lab) + num_special_tokens)
                                   for lab in human_transcript_labels]
        # CTC doesn't use SOS nor EOS; LAS doesn't use EPS but add anyway.
        eps_index, sos_index, eos_index, pad_index = 0, 1, 2, 3

        if append_eos_token:
            # Insert an EOS token to the end of all the labels.
            # This is important for the LAS model.
            human_transcript_labels_ = []
            for i in range(len(human_transcript_labels)):
                new_label_i = human_transcript_labels[i] + [eos_index]
                human_transcript_labels_.append(new_label_i)
            human_transcript_labels = human_transcript_labels_
        self.human_transcript_labels = human_transcript_labels

        # Include epsilon, SOS, and EOS tokens.
        self.num_class = len(VOCAB) + len(SILENT_VOCAB) + num_special_tokens
        self.num_labels = self.num_class
        self.eps_index = eps_index
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.indices = indices

    def split_data(self, valid_indices, train_ratio=0.8, val_ratio=0.1):
        """Splits data into train, val, and test sets based on speaker. When
    evaluating methods on the test split, we measure how well they generalize
    to new (unseen) speakers.

    Concretely, this stores and returns indices belonging to each split.
    """
        rs = np.random.RandomState(42)

        speaker_ids = self.label_data['speaker_ids']
        unique_speaker_ids = sorted(list(set(speaker_ids)))
        unique_speaker_ids = np.array(unique_speaker_ids)

        # Shuffle so the speaker IDs are distributed.
        rs.shuffle(unique_speaker_ids)

        num_speaker = len(unique_speaker_ids)
        num_train = int(train_ratio * num_speaker)
        num_val = int(val_ratio * num_speaker)
        num_test = num_speaker - num_train - num_val

        train_speaker_ids = unique_speaker_ids[:num_train]
        val_speaker_ids = unique_speaker_ids[num_train:num_train + num_val]
        test_speaker_ids = unique_speaker_ids[num_train + num_val:]

        train_speaker_dict = dict(zip(train_speaker_ids, ['train'] * num_train))
        val_speaker_dict = dict(zip(val_speaker_ids, ['val'] * num_val))
        test_speaker_dict = dict(zip(test_speaker_ids, ['test'] * num_test))
        # unpack dicts into one
        speaker_dict = {**train_speaker_dict, **val_speaker_dict,
                        **test_speaker_dict}

        train_indices, val_indices, test_indices = [], [], []
        for i in range(len(speaker_ids)):
            speaker_id = speaker_ids[i]
            if speaker_dict[speaker_id] == 'train':
                train_indices.append(i)
            elif speaker_dict[speaker_id] == 'val':
                val_indices.append(i)
            elif speaker_dict[speaker_id] == 'test':
                test_indices.append(i)
            else:
                raise Exception('split not recognized.')

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        # Make sure to only keep "valid indices" i.e. those with more than 4
        # words in the transcription.
        # find common elements between two arrays and update to only contain common elements
        train_indices = np.intersect1d(train_indices, valid_indices)
        val_indices = np.intersect1d(val_indices, valid_indices)
        test_indices = np.intersect1d(test_indices, valid_indices)

        return train_indices, val_indices, test_indices

    def get_primary_task_data(self, index):
        """Returns audio and transcript information for a single utterance.

    Args:
      index: Index of an utterance.

    Returns:
      log melspectrogram, wav length, transcript label, transcript length
    """
        input_feature = None
        input_length = None
        human_transcript_label = None
        human_transcript_length = None

        wav = self.waveform_data[f'{index}'][:]  # An h5py file uses string keys.
        transcript = self.human_transcript_labels[index]
        sr = 8000  # We fix the sample rate for you.
        mfcc = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
                                              win_length=self.win_length, n_mels=self.n_mels)
        # add 1e-7 to avoid log of zero
        norm_mfcc = librosa.util.normalize(np.log(mfcc + 1e-7))
        # transpose since model expects input shape (T, n_mels)
        norm_mfcc = norm_mfcc.T
        input_feature, input_length = pad_wav(norm_mfcc, self.wav_max_length)
        input_feature = torch.from_numpy(input_feature).float()
        human_transcript_label, human_transcript_length = pad_transcript_label(transcript, self.transcript_max_length,
                                                                               pad=self.pad_index)

        return input_feature, input_length, human_transcript_label, human_transcript_length

    def load_waveforms(self):
        # Make a file pointer to waveforms file.
        waveform_h5 = h5py.File(os.path.join(self.root, 'data.h5'), 'r')
        self.waveform_data = waveform_h5.get('waveforms')

    def __getitem__(self, index):
        """Serves primary task data for a single utterance."""
        if not hasattr(self, 'waveform_data'):
            # Do this in __getitem__ function so we enable multiprocessing.
            self.load_waveforms()
        index = int(self.indices[index])
        return self.get_primary_task_data(index)

    def __len__(self):
        """Returns total number of utterances in the dataset."""
        return len(self.indices)

