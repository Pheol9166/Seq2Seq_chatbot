import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, data, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[1]
        max_len = target.shape[0]
        vocab_size = self.decoder.ouput_dim

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(data.device)
        encoder_outputs, hidden = self._encoder(data)

        inputs = target[0, :]

        for t in range(1, max_len):
            outputs, hidden = self._decoder(inputs, hidden, encoder_outputs)
            outputs[t] = outputs
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = outputs.argmax(1)
            inputs = target[t] if teacher_force else top1

        return outputs