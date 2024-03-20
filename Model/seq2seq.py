import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):

  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()
    self._encoder = encoder
    self._decoder = decoder

  def forward(self, data, target, teacher_forcing_ratio=0.5):
    batch_size, max_len = target.shape
    vocab_size = self._decoder._output_dim

    outputs = torch.zeros(max_len, batch_size, vocab_size).to(data.device)
    _, hidden, cell = self._encoder(data)

    inputs = torch.full((batch_size, ), 1, device=data.device)

    for t in range(0, max_len):
      output, hidden, cell = self._decoder(inputs, hidden, cell)
      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.argmax(1)
      inputs = target[:, t] if teacher_force else top1

    return outputs.permute(1, 0, 2)
