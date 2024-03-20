import torch.nn as nn


class Encoder(nn.Module):

  def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=1):
    super(Encoder, self).__init__()
    self._embedding = nn.Embedding(input_dim, embedding_dim)
    self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)

  def forward(self, x):
    # x: (batch_size, seq_len)
    embed = self._embedding(x).permute(1, 0, 2)
    outputs, (hidden_state, cell_state) = self._lstm(embed)
    # outputs: (seq_len, batch_size, hidden_dim)
    # hidden_state: [hidden state, cell state] (Bidirectional x number of layers, batch_size, hidden_size)
    return outputs, hidden_state, cell_state
