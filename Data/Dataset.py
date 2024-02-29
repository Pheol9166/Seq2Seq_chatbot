import torch
from torch.utils.data import Dataset
from konlpy.tag import Okt
from utils.TextTokenizer import TextTokenizer
from utils.text_preprocessing import tokenize_text

class TextDataset(Dataset):
    def __init__(self, text, label, tokenizer, max_len):
        self._text = text
        self._label = label
        self._tokenizer: TextTokenizer = tokenizer
        self._tagger = Okt()
        self._max_len = max_len

    def __len__(self):
        return len(self._text)
    
    def __getitem__(self, index):
        text = tokenize_text(self._text[index], self._tagger)
        label = tokenize_text(self._label[index], self._tagger)
        sequence = self._tokenizer.text_to_sequence(text)
        padded_sequence = self._tokenizer.padding(sequence, maxlen=self._max_len)

        return {
            'inputs': torch.tensor(padded_sequence, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }       