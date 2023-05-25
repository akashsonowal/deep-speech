import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from comet_ml import Experiment

from deep_speech import (
    TextTransform,
    data_processing,
    SpeechRecognitionModel,
    GreedyDecoder,
)
from util import IterMeter, cer, wer


def train(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    epoch,
    iter_meter,
    experiment,
):
    model.train()
    data_len = len(train_loader.dataset)
    with experiment.train():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            experiment.log_metric("loss", loss.item(), step=iter_meter.get())
            experiment.log_metric(
                "learning_rate", scheduler.get_lr(), step=iter_meter.get()
            )

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(spectrograms),
                        data_len,
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )


def test(model, device, test_loader, criterion, epoch, iter_meter, experiment):
    print("\n Evaluating")
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with experiment.test():
        with torch.no_grad():
            for I, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)

                decoded_preds, decoded_targets = GreedyDecoder(
                    output.transpose(0, 1), labels, label_lengths
                )
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
        avg_cer = sum(test_cer) / len(test_cer)
        avg_wer = sum(test_wer) / len(test_wer)

        experiment.log_metric("test_loss", test_loss, step=iter_meter.get())
        experiment.log_metric("cer", avg_cer, step=iter_meter.get())
        experiment.log_metric("wer", avg_wer, step=iter_meter.get())

        print(
            "Test set: Average loss: {:.4f}, Average CER: {:.4f} Average WER: {:.4f}\n".format(
                test_loss, avg_cer, avg_wer
            )
        )


def main(
    learning_rate=5e-4,
    batch_size=20,
    epochs=10,
    train_url="train-clean-100",
    test_url="test-clean",
    experiment=Experiment(api_key="<Your API Key>", disabled=True),
):
    h_params = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    experiment.log_parameters(h_params)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    train_dataset = torchaudio.datasets.LIBRISPEECH(
        "./data", url=train_url, download=True
    )
    test_dataset = torchaudio.datasets.LIBRISPEECH(
        "./data", url=test_url, download=True
    )

    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35),
    )
    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
    text_transform = TextTransform()

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=h_params["batch_size"],
        shuffle=True,
        collate_fn=lambda x: data_processing(
            x, train_audio_transforms, text_transform, **kwargs
        ),
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=h_params["batch_size"],
        shuffle=False,
        collate_fn=lambda x: data_processing(
            x, valid_audio_transforms, text_transform, **kwargs
        ),
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

    print(model)
    print(
        "Num model parameters", sum([param.nelement() for param in model.parameters()])
    )

    criterion = nn.CTCLoss(blank=28).to(device)
    optimizer = optim.AdamW(model.parameters(), h_params["learning_rate"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=h_params["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=h_params["epochs"],
        anneal_strategy="linear",
    )

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


if __name__ == "__main__":
    main()
