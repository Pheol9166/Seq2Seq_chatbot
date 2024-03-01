import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_processing import load_data, create_dataset, create_dataloader, split_data
from Model.Encoder import Encoder
from Model.Decoder import Decoder
from Model.Seq2Seq import Seq2Seq

DATA_PATH = "./data/raw_data/ChatbotData.csv"
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = load_data(DATA_PATH)
questions = data['Q']
answers = data['A']

dataset = create_dataset(questions, answers, max_len=25)
train_dataset, test_dataset = split_data(dataset, test_size=0.2)
train_loader = create_dataloader(train_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

INPUT_DIM = len(dataset.tokenizer.vocab)
OUTPUT_DIM = len(dataset.tokenizer.vocab)
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.3
EPOCHS = 10
LEARNING_RATE = 0.001

encoder = Encoder(input_dim=INPUT_DIM,
                  embedding_dim=EMBEDDING_DIM,
                  hidden_dim=HIDDEN_DIM,
                  num_layers=NUM_LAYERS).to(device)
decoder = Decoder(output_dim=OUTPUT_DIM,
                  embedding_dim=EMBEDDING_DIM,
                  hidden_dim=HIDDEN_DIM,
                  num_layers=NUM_LAYERS,
                  dropout=DROPOUT).to(device)
seq = Seq2Seq(encoder, decoder).to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(seq.parameters(), lr=LEARNING_RATE)


def train(model, train_loader, criterion, optimizer, device):
  model.train()
  running_loss = 0

  for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data, target)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  return running_loss / len(train_loader)


for epoch in range(EPOCHS):
  train_loss = train(seq, train_loader, loss, optimizer, device)
  print(f"Epoch {epoch + 1} / {EPOCHS} : Loss {train_loss}")
