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
    self._output_dim = output_dim
    self._embedding = nn.Embedding(output_dim, embedding_dim)
    self._dropout = dropout
    self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
    self._fc_out = nn.Linear(hidden_dim, output_dim)

  def forward(self, x, hidden_state, cell_state):
    # x: 현재 시점의 입력 토큰 (shape: [batch_size])
    # hidden_state: 이전 시점의 은닉 상태 (shape: [num_layers * num_directions, batch_size, hidden_dim])
    x = x.unsqueeze(0)
    embed = self._embedding(x) # shape: [seq_length]
    embed = F.dropout(embed, p=self._dropout)
    output, (hidden_state, cell_state) = self._lstm(embed,
                                                    (hidden_state, cell_state))
    # output: [seq_length, output_dim]
    output = self._fc_out(output.squeeze(0))

    return output, hidden_state, cell_state
