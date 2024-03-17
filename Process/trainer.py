import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.hyperparameter import Hyperparmeters
from utils.config import load_config
from utils.data_processing import create_dataloader
from utils.resource import ModelResource
from Model.encoder import Encoder
from Model.decoder import Decoder
from Model.seq2seq import Seq2Seq



class Trainer:

  def __init__(self, dataset, config_path):
    self._dataset = dataset
    self._config = config_path
    self.train_loss_list = []
    self.test_loss_list = []
    self.initialize_model()

  def initialize_model(self):
    config = load_config(self._config)

    self.hyperparams = Hyperparmeters()
    self.hyperparams.set_from_dict(config)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.vocab_size = len(self._dataset._tokenizer.vocab)

    encoder = Encoder(input_dim=self.vocab_size,
                      embedding_dim=self.hyperparams.get_embedding_dim(),
                      hidden_dim=self.hyperparams.get_hidden_dim(),
                      num_layers=self.hyperparams.get_num_layers()).to(
                          self.device)
    decoder = Decoder(output_dim=self.vocab_size,
                      embedding_dim=self.hyperparams.get_embedding_dim(),
                      hidden_dim=self.hyperparams.get_hidden_dim(),
                      num_layers=self.hyperparams.get_num_layers(),
                      dropout=self.hyperparams.get_dropout()).to(self.device)
    self.model = Seq2Seq(encoder, decoder).to(self.device)

    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(),
                                lr=self.hyperparams.get_learning_rate())

    train_dataset, test_dataset = self._dataset.split()
    self.train_loader = create_dataloader(train_dataset,
                                          self.hyperparams.get_batch_size())
    self.test_loader = create_dataloader(test_dataset,
                                         self.hyperparams.get_batch_size())

  def train_epoch(self):
    self.model.train()
    running_loss = 0

    for data, target in self.train_loader:
      data, target = data.to(self.device), target.to(self.device)

      self.optimizer.zero_grad()

      output = self.model(data, target)
      output_dim = output.size(2)

      output = output.reshape(-1, output_dim)
      # (batch_size * seq_len, vocab_len)
      target = target.reshape(-1)  # (batch_size * seq_len)

      loss = self.criterion(output, target)
      loss.backward()
      self.optimizer.step()

      running_loss += loss.item() * data.size(0)
      # noramlization dividing loss by batch_size

    train_loss = running_loss / len(self.train_loader)
    self.train_loss_list.append(train_loss)

    return train_loss

  def evaluate(self):
    with torch.no_grad():
      self.model.eval()
      test_loss = 0

      for data, target in self.test_loader:
        data, target = data.to(self.device), target.to(self.device)

        output = self.model(data, target)
        output_dim = output.size(2)

        output = output.reshape(-1, output_dim)
        # (batch_size * seq_len, vocab_len)
        target = target.reshape(-1)  # (batch_size * seq_len)

        loss = self.criterion(output, target)
        test_loss += loss.item() * data.size(0)
        # noramlization dividing loss by batch_size

    eval_loss = test_loss / len(self.test_loader)
    self.test_loss_list.append(eval_loss)

    return eval_loss

  def train(self):
    for epoch in range(self.hyperparams.get_epochs()):
      train_loss = self.train_epoch()
      eval_loss = self.evaluate()
      print(
          f"Epoch {epoch + 1} / {self.hyperparams.get_epochs()} : train_loss {train_loss}, eval_loss {eval_loss}"
      )

  def loss_graph(self):
    plt.plot(self.train_loss_list, 'b')
    plt.plot(self.test_loss_list, 'orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.legend(('Train Loss', 'Test Loss'))
    plt.show()

  def tune_hyperparams(self):
    pass

  def save_model(self):
    mr = ModelResource(self.model, self._dataset._tokenizer, self.optimizer,
                       self.criterion, self.device)
    mr.save(self._config["MODEL_PATH"])

  def load_model(self):
    mr = ModelResource.load(self._config["MODEL_PATH"])
    self.model = mr.model
    self.optimizer = mr.optimizer
    self.criterion = mr.criterion
    self.device = mr.device
