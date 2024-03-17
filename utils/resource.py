import torch
from Model.encoder import Encoder
from Model.decoder import Decoder
from Model.seq2seq import Seq2Seq
from utils.tokenizer import TextTokenizer


class ModelResource:

  def __init__(self, model, tokenizer, optimizer, criterion, device):
    self.model: Seq2Seq = model
    self.tokenizer: TextTokenizer = tokenizer
    self.optimizer: torch.optim.Adam = optimizer
    self.criterion: torch.nn.CrossEntropyLoss = criterion
    self.device = device

  def save(self, file_path: str) -> None:
    checkpoint = {
        'encoder_state_dict': self.model._encoder.state_dict(),
        'decoder_state_dict': self.model._decoder.state_dict(),
        'tokenizer': self.tokenizer,
        'optimizer_state_dict': self.optimizer.state_dict(),
        'criterion_state_dict': self.criterion.state_dict(),
        'device': self.device
    }
    torch.save(checkpoint, file_path)

  @classmethod
  def load(cls, file_path):
    checkpoint = torch.load(file_path)
    encoder = Encoder(input_dim=100, embedding_dim=256, hidden_dim=512)
    decoder = Decoder(output_dim=100, embedding_dim=256, hidden_dim=512)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    model = Seq2Seq(encoder, decoder)

    tokenizer = checkpoint['tokenizer']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = torch.nn.CrossEntropyLoss()
    loss.load_state_dict(checkpoint['criterion_state_dict'])
    device = checkpoint['device']

    return cls(model, tokenizer, optimizer, loss, device)
