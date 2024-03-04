import torch
from torch.utils.data import Dataset
from utils.TextTokenizer import TextTokenizer


class TextDataset(Dataset):

  def __init__(self, text, label, max_len):
    super(TextDataset, self).__init__()
    self._tokenizer: TextTokenizer = TextTokenizer(text, label)
    self._text = self._tokenizer.padding(
        self._tokenizer.text_to_sequence(text), maxlen=max_len)
    self._label = self._tokenizer.padding(
        self._tokenizer.text_to_sequence(label), maxlen=max_len)
    self._max_len = max_len

  def __len__(self):
    return len(self._text)

  def __getitem__(self, index):
    text = self._text[index]
    label = self._label[index]

    return torch.tensor(text, dtype=torch.long), torch.tensor(label,
                                                              dtype=torch.long)
