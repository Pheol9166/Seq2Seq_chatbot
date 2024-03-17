class Hyperparmeters:

  def __init__(self,
               embedding_dim=256,
               hidden_dim=512,
               learning_rate=0.001,
               batch_size=64,
               epochs=10,
               num_layers=1,
               dropout=0.2):
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.epochs = epochs
    self.num_layers = num_layers
    self.dropout = dropout

  def __repr__(self):
    return f'Hyperparameters(embedding_dim={self.embedding_dim}, hidden_dim={self.hidden_dim}, learning_rate={self.learning_rate}, batch_size={self.batch_size}, epochs={self.epochs}, num_layers={self.num_layers}, dropout={self.dropout})'

  def set_embedding_dim(self, embedding_dim):
    self.embedding_dim = embedding_dim

  def set_hidden_dim(self, hidden_dim):
    self.hidden_dim = hidden_dim

  def set_learning_rate(self, learning_rate):
    self.learning_rate = learning_rate

  def set_batch_size(self, batch_size):
    self.batch_size = batch_size

  def set_epochs(self, epochs):
    self.epochs = epochs

  def set_num_layers(self, num_layers):
    self.num_layers = num_layers

  def set_dropout(self, dropout):
    self.dropout = dropout

  def set_from_dict(self, hyperparams_dict):
    self.embedding_dim = hyperparams_dict.get("EMBEDDING_DIM",
                                              self.embedding_dim)
    self.hidden_dim = hyperparams_dict.get("HIDDEN_DIM", self.hidden_dim)
    self.learning_rate = hyperparams_dict.get("LEARNING_RATE",
                                              self.learning_rate)
    self.batch_size = hyperparams_dict.get("BATCH_SIZE", self.batch_size)
    self.epochs = hyperparams_dict.get("EPOCHS", self.epochs)
    self.num_layers = hyperparams_dict.get("NUM_LAYERS", self.num_layers)
    self.dropout = hyperparams_dict.get("DROPOUT", self.dropout)

  def get_embedding_dim(self):
    return self.embedding_dim

  def get_hidden_dim(self):
    return self.hidden_dim

  def get_learning_rate(self):
    return self.learning_rate

  def get_batch_size(self):
    return self.batch_size

  def get_epochs(self):
    return self.epochs

  def get_num_layers(self):
    return self.num_layers

  def get_dropout(self):
    return self.dropout
