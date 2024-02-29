import re
import urllib.request
from utils.WordVocab import WordVocab
from konlpy.tag import Okt

def cleaning_text(text: str) -> str:
    """_Cleaning text by removing special characters and numbers_

    Args:
        text (str): _text to clean_

    Returns:
        str: _cleaned text result_
    """
    return re.sub(r'[^가-힣\s]', '', text)

def remove_stopwords(tokenized_text: list[list[str]]) -> str:
    """_Removing stopwords from text_

    Args:
        tokenized_text (list[list[str]]): _tokenized_text to remove stowords_

    Returns:
        str: _removed result_
    """
    
    url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ko/master/raw/gh-stopwords-json-ko.txt"
    urllib.request.urlretrieve(url=url, filename="stopwords.txt")

    with open("stopwords.txt", "r") as fr:
        stopwords = [word.strip() for word in fr.readlines()]

    return [token for sent in tokenized_text for token in sent if token not in stopwords]

def tokenize_text(text: str, tokenizer: Okt) -> list:
    """_Tokenize text using tokenizer_

    Args:
        text (str): _text to tokenize_
        tokenizer (Okt): _Okt tokenizer object from Konlpy_

    Returns:
        list: _tokenized result_
    """
    return tokenizer.tokenize(text)

def build_vocab(tokenized_text: list[list[str]]) -> WordVocab:
    """_build Vocab from tokens_

    Args:
        tokenized_text (list[list[str]]): _description_

    Returns:
        WordVocab: _description_
    """
    vocab = WordVocab()

    for sent in tokenized_text:
        for token in sent:
            vocab.add_word(token)
    
    return vocab

def load_vocab(filename: str) -> WordVocab:
    """_load Vocab from file_

    Args:
        filename (str): _name of file_

    Returns:
        WordVocab: _loaded Vocab_
    """
    vocab = WordVocab()
    with open(filename, 'r') as fr:
        for line in fr.readline():
            word, idx, count = line.strip().split('\t')
            idx = int(idx)
            count = int(count)
            vocab.word2idx[word] = idx
            vocab.idx2word[idx] = word
            vocab.count[word] = count
            vocab.idx = max(vocab.idx, idx + 1)
    
    return vocab



