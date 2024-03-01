import torch.nn as nn


class Encoder(nn.Module):

  def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=1):
    super(Encoder, self).__init__()
    self._embedding = nn.Embedding(input_dim, embedding_dim)
    self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)

  def forward(self, x):
    embed = self._embedding(x)
    outputs, hidden_state = self._lstm(embed)

    return hidden_state
