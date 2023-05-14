import os
import torch
from torch import nn
import torchaudio
from comet_ml import Experiment
import numpy as np

train_dataset = torchaudio.datasets.LIBRISPEECH(
    "./", url="train-clean-100", download=True
)
test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
)
valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

optimizer = optim.AdamW(model.parameters(), hparams["learning_rate"])
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=hparams["learning_rate"],
    steps_per_epoch=int(len(train_loader)),
    epochs=hparams["epochs"],
    anneal_startegy="linear",
)
criterion = nn.CTCLoss(blank=28).to(device)


def train():
    pass


def test():
    pass


def main():
    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        train(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            iter_meter,
            experiment,
        )
        test(
            model,
            device,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            iter_meter,
            experiment,
        )
