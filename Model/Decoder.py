import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

  def __init__(self,
               output_dim,
               embedding_dim,
               hidden_dim,
               dropout=0.2,
               num_layers=1):
    super(Decoder, self).__init__()
    self._embedding = nn.Embedding(output_dim, embedding_dim)
    self._dropout = dropout
    self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
    self._fc_out = nn.Linear(hidden_dim, output_dim)

  def forward(self, x, hidden_state):
    x = x.unsqueeze(0)
    embed = self._embedding(x)
    embed = F.dropout(embed, p=self._dropout)
    output, hidden_state = self._lstm(embed, hidden_state)
    output = self._fc_out(output.squeeze(0))

    return output, hidden_state
