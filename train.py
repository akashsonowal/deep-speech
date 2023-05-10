def main():
  iter_meter = IterMeter()
  for epoch in range(1, epochs + 1):
    train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
    test(model, device, test_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
