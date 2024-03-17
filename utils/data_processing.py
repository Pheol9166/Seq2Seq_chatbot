import pickle
import pandas as pd
from Data.dataset import TextDataset
from torch.utils.data import DataLoader


def load_data(file_path: str) -> pd.DataFrame:
  return pd.read_csv(file_path, encoding='utf-8', sep=',')


def load_dataset(file_path: str) -> TextDataset:
  with open(file_path, 'rb') as fr:
    dataset = pickle.load(fr)
  return dataset


def create_dataset(data: pd.Series, label: pd.Series,
                   max_len: int) -> TextDataset:
  return TextDataset(data, label, max_len)


def create_dataloader(dataset: TextDataset,
                      batch_size: int,
                      shuffle: bool = True) -> DataLoader:
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
