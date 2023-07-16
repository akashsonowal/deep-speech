import os
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import torchaudio

from deep_speech import (
    TextTransform,
    data_processing,
    SpeechRecognitionModel,
    GreedyDecoder,
)


def asr(data):
    h_params = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    audio_transforms = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=128
    )
    text_transform = TextTransform()

    spectrograms, labels, input_lengths, label_lengths = data_processing(
        data, audio_transforms, text_transform
    )

    model = SpeechRecognitionModel(
        h_params["n_cnn_layers"],
        h_params["n_rnn_layers"],
        h_params["rnn_dim"],
        h_params["n_class"],
        h_params["n_feats"],
        h_params["stride"],
        h_params["dropout"],
    ).to(device)

    model.load_state_dict(torch.load("./model_checkpoint/deep_speech.pth"))
    model.eval()
    output = model(spectrograms)
    output = F.log_softmax(output, dim=2)
    decoded_preds, decoded_targets = GreedyDecoder(
        output, labels, label_lengths, text_transform
    )

    print("The original labels is", decoded_targets)
    print("**********")
    print("The transcription by the model is", decoded_preds)


if __name__ == "__main__":
    if os.listdir("./data"):
        data = torchaudio.datasets.LIBRISPEECH(
            "./data", url="test-clean", download=False
        )
        asr(data)
    else:
        raise ValueError("upload audio files for transcribing")