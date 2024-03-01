import torch
from torch.utils.data import Dataset
from utils.TextTokenizer import TextTokenizer


class TextDataset(Dataset):

  tokenizer: TextTokenizer = TextTokenizer()

  def __init__(self, text, label, max_len):
    super(TextDataset, self).__init__()
    self._text = text
    self._label = label
    self._max_len = max_len

  def __len__(self):
    return len(self._text)

  def __getitem__(self, index):
    text = self._text[index]
    label = self._label[index]
    text_sequence = TextDataset.tokenizer.text_to_sequence(text)
    padded_text = TextDataset.tokenizer.padding(text_sequence,
                                                maxlen=self._max_len)
    label_sequence = TextDataset.tokenizer.text_to_sequence(label)
    padded_label = TextDataset.tokenizer.padding(label_sequence,
                                                 maxlen=self._max_len)

    return {
        'inputs': torch.tensor(padded_text, dtype=torch.long),
        'labels': torch.tensor(padded_label, dtype=torch.long)
    }
