import librosa
import torch
from utils import pad_wav


def run_inference(
        model, wav, device=None, sr=8000, n_mels=128, n_fft=256, win_length=256,
        hop_length=128, wav_max_length=512, labels=None, label_lengths=None):
    """Run model on a .WAV file and returns a string utterance.

  Args:
    system: a pl.LightningModule for your chosen model.
    wav: a .WAV file of an utterance
    device: GPU -> torch.device('cuda')

  Returns:
    A string for the utterance transcribed by your model.
    :param label_lengths: true length of unpadded character labels
    :param labels: padded character labels
    :param wav_max_length: integer maximum number of timesteps in a waveform
    :param hop_length: integer number of frames to skip in between
    :param win_length: integer should be <= n_fft
    :param n_fft: integer number of fourier components
    :param n_mels:
    :param sr: sampling rate
    :param device: gpu or cpu
    :param wav: wav file
    :param model: CTC, LAS or LASCTC
  """
    input_feature = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, n_fft=n_fft, win_length=win_length,
        hop_length=hop_length)
    input_feature = librosa.util.normalize(np.log(input_feature + 1e-7))
    input_feature = input_feature.T
    input_feature, input_length = pad_wav(input_feature, wav_max_length)
    input_feature = torch.from_numpy(input_feature).unsqueeze(0)
    input_lengths = torch.LongTensor([input_length])
    if device is not None:
        input_feature = input_feature.to(device)
        input_lengths = input_lengths.to(device)
        if labels is not None:  # to test teacher-forcing
            labels = labels.to(device)
            labels_lengths = label_lengths.to(device)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        log_probs, embeddings = model(input_feature, input_lengths)
        hypotheses, hypothesis_lengths, references, reference_length = \
            model.decode(
                log_probs, input_lengths, labels, label_lengths,
                model.train_dataset.sos_index,
                model.train_dataset.eos_index,
                model.train_dataset.pad_index,
                model.train_dataset.eps_index)

        utterance = model.train_dataset.indices_to_chars(hypotheses)
    return utterance

# load a model from checkpoint
# ctc_model_path = os.path.join(MODEL_PATH, 'ctc/epoch=19-step=1620.ckpt')
# model = LightningCTC.load_from_checkpoint()
# run_inference


