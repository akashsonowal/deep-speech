import torchaudio

train_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

torchaudio.transforms.FrequenceMasking()
torchaudio.transforms.TimeMasking()



def main():
  iter_meter = IterMeter()
  for epoch in range(1, epochs + 1):
    train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
    test(model, device, test_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
