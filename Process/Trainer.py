import torch
import math
import pickle
import matplotlib.pyplot as plt
from Model.ModelResource import ModelResource


class Trainer:

  def __init__(self, model, optimizer, criterion, device):
    self._model = model
    self._optimizer = optimizer
    self._criterion = criterion
    self._device = device
    self.loss_list = []

  def train_epoch(self, train_loader):
    print(self._model)
    self._model.train()
    running_loss = 0

    for data, target in train_loader:
      data, target = data.to(self._device), target.to(self._device)

      self._optimizer.zero_grad()

      output = self._model(data, target)
      output_dim = output.size(2)

      output = output.reshape(-1,
                              output_dim)  # (batch_size * seq_len, vocab_len)
      target = target.reshape(-1)  # (batch_size * seq_len)

      loss = self._criterion(output, target)
      loss.backward()
      self._optimizer.step()

      running_loss += loss.item()
      self.loss_list.append(running_loss)

    return running_loss / len(train_loader)

  def evaluate(self, test_loader):
    with torch.no_grad():
      self._model.eval()
      test_loss = 0

      for data, target in test_loader:
        data, target = data.to(self._device), target.to(self._device)

        output = self._model(data, target)
        output_dim = output.size(2)

        output = output.reshape(
            -1, output_dim)  # (batch_size * seq_len, vocab_len)
        target = target.reshape(-1)  # (batch_size * seq_len)

        loss = self._criterion(output, target)
        test_loss += loss.item()

    return test_loss / len(test_loader)

  def train(self, train_loader, test_loader, epochs=10):
    for epoch in range(epochs):
      train_loss = self.train_epoch(train_loader)
      perplexity = self.evaluate(test_loader)
      print(
          f"Epoch {epoch + 1} / {epochs} : Loss {train_loss}, Perplexity {perplexity}"
      )

  def loss_graph(self):
    plt.plot(self.loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.show()