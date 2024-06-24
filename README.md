# END to END Automatic Speech Recognition

This project implements three different end-to-end Automatic Speech Recognition (ASR) architectures using PyTorch.

The ASR models are based on the following architectures:
- CTC<sup>[1](#Reference)</sup>
- Listen, Attend and Spell (LAS)<sup>[2](#Reference)</sup>
- LAS-CTC<sup>[3](#Reference)</sup>

The models were trained and tested on a subset of the HarperValleyBank Dataset<sup>[4](#Reference)</sup>, which is hosted [here](http://web.stanford.edu/class/cs224s/download/harpervalleybank.zip). The dataset is used to train models that predict each spoken character.

## Highlights

### Feature Extraction
- Uses Librosa to extract WAV log melspectrogram
- Character encoding

### Training End-to-End ASR
- Multiple implementations of ASR model architectures, including attention-based models
- Regularization of attention-based networks to respect CTC alignments (LAS-CTC)
- Utilizes Lightning Trainer API
- Training process logs and visualizations with [Wandb](https://wandb.ai/site)
- Teacher-forcing

### Decoding
- Greedy decoding
- Imposes a CTC objective on the decoding
- CTC-Rules

## Getting Started

1. Download and unzip the dataset:
    ```bash
    unzip harper_valley_bank_minified.zip -d data
    ```

## Model Run Report

![ASR_model_report.png](data/ASR_model_report.png)
<p style="text-align: center;">Model run report obtained from Wandb</p>

## Reference
1. [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf), A Graves *et al*.
2. [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211v2), W Chan *et al.*
3. [Joint CTC-Attention based End-to-End Speech Recognition using Multi-task Learning](https://arxiv.org/abs/1609.06773), S Kim *et al.*
4. [CS224S: Spoken Language Processing](https://web.stanford.edu/class/cs224s/)

This README provides an overview of the project, highlighting its main features, the technology stack, and usage instructions. For more detailed documentation, please refer to the project files and comments within the code.
