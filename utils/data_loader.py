import pandas as pd
import pickle
from Data.Dataset import TextDataset
from utils.TextTokenizer import TextTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, encoding='utf-8')

def load_dataset(file_path: str) -> TextDataset:
    with open(file_path, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset

def create_dataset(data: pd.Series, label: pd.Series, tokenizer: TextTokenizer, max_len: int) -> TextDataset:
    return TextDataset(data, label, tokenizer, max_len)

def create_dataloader(dataset: TextDataset, batch_size: int, shuffle: bool=True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def save_dataset(dataset: TextDataset, file_path: str) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)

