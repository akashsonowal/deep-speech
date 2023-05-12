import torchaudio

train_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

train_audio_transforms = nn.Sequential(
  torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
  torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
  torchaudio.transforms.TimeMasking(time_mask_param=35)
)
valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

def main():
  iter_meter = IterMeter()
  for epoch in range(1, epochs + 1):
    train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
    test(model, device, test_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
