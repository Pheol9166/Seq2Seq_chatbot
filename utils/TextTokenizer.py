import re
import pandas as pd
import urllib.request
from scipy.sparse import spmatrix
from konlpy.tag import Okt
from utils.WordVocab import WordVocab
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextTokenizer:

  def __init__(self, data: pd.Series, label: pd.Series):
    self.data = data
    self.label = label
    self.tagger = Okt()
    self.vocab = WordVocab()
    self.vectorizer = None

  @staticmethod
  def clean_text(text: str) -> str:
    """_Cleaning text by removing special characters and numbers_
    
      Args:
          text (str): _text to clean_
    
      Returns:
          str: _cleaned text result_
    """
    return re.sub(r'[^가-힣\s]', '', text)

  @staticmethod
  def remove_stopwords(tokenized_text: list[str]) -> list[str]:
    """_Removing stopwords from text_
  
      Args:
          tokenized_text (list[str]): _tokenized_text to remove stowords_
  
      Returns:
          list[str]: _removed result_
    """

    url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ko/master/raw/gh-stopwords-json-ko.txt"
    urllib.request.urlretrieve(url=url, filename="stopwords.txt")

    with open("stopwords.txt", "r") as fr:
      stopwords = [word.strip() for word in fr.readlines()]

    return [token for token in tokenized_text if token not in stopwords]

  def tokenize_text(self, text: str) -> list[str]:
    """_Tokenize text using tokenizer_

      Args:
          text (str): _text to tokenize_
          tokenizer (Okt): _Okt tokenizer object from Konlpy_

      Returns:
          list[str]: _tokenized result_
    """
    return self.tagger.morphs(text)

  def preprocess_text(self, text: str) -> list[str]:
    return TextTokenizer.remove_stopwords(
        self.tokenize_text(TextTokenizer.clean_text(text)))

  def build_vocab(self, samples: pd.Series, labels: pd.Series) -> None:
    """_build Vocab from tokens_
  
      Args:
          tokenized_text (list[list[str]]): _description_
  
      Returns:
          WordVocab: _description_
    """

    samples = samples.map(self.preprocess_text)
    labels = labels.map(self.preprocess_text)

    for x, y in zip(samples, labels):
      for x_tok, y_tok in zip(x, y):
        self.vocab.add_word(x_tok)
        self.vocab.add_word(y_tok)

  def load_vocab(self, filename: str) -> None:
    """_load Vocab from file_

      Args:
          filename (str): _name of file_

      Returns:
          WordVocab: _loaded Vocab_
    """
    with open(filename, 'r') as fr:
      for line in fr.readline():
        word, idx, count = line.strip().split('\t')
        idx = int(idx)
        count = int(count)
        self.vocab.word2idx[word] = idx
        self.vocab.idx2word[idx] = word
        self.vocab.count[word] = count
        self.vocab.idx = max(self.vocab.idx, idx + 1)

  def text_to_sequence(self,
                       texts: list[str],
                       vocab_file: str | None = None) -> list[list[int]]:
    """_integer encode using WordVocab_

        Args:
            text (list[str]): _text to encode_
        Returns:
            list[list[int]]: _result of integer encoding_
        """
    if vocab_file is None:
      self.build_vocab(self.data, self.label)
    else:
      self.load_vocab(vocab_file)

    tokenized_texts = [self.preprocess_text(sent) for sent in texts]

    sequences = []
    for sent in tokenized_texts:
      sent = ['<SOS>'] + sent + ['<EOS>']
      sequence = [
          self.vocab.word2idx.get(token, self.vocab.word2idx['<UNK>'])
          for token in sent
      ]
      sequences.append(sequence)
    return sequences

  def fit_on_texts(self,
                   text: str | list[list[str]],
                   mode: str = 'binary') -> None:
    """_fit vectorizer based on tokenized text_

        Args:
            text (str | list[list[str]]): _text for train_
            mode (str, optional): _option of vectorizer, 'binary' gets sklearn's CountVectorizer and 'tfidf' gets sklearn's TfidfVectorizer_. Defaults to 'binary'.
        """
    if mode == 'binary':
      self.vectorizer = CountVectorizer(binary=True)
    elif mode == 'tfidf':
      self.vectorizer = TfidfVectorizer()
    else:
      raise ValueError("Invalid mode.")

    self.vectorizer.fit(text)

  def transform_to_matrix(self, text: list[list[str]]) -> spmatrix:
    """_transform text to matrix_

        Args:
            text (list[list[str]]): _text to transform_

        Returns:
            spmatrix: _result of vectorization_
        """
    if self.vectorizer is None:
      raise ValueError("Tokenizer is None. Call fit_on_texts() first")
    else:
      return self.vectorizer.transform(text)

  def padding(
      self,
      encoded_text: list[list[int]],
      maxlen: int = 3,
  ) -> list[list[int]]:
    """_Pad the encoded text using Keras's pad_sequences func_

        Args:
            encoded_text (list[list[int]]): _encoded text to be padded_
            maxlen (int, optional): _maximum length of sequences after padding_. Defaults to 3.

        Returns:
            list[list[int]]: _padded sequences_
        """
    return pad_sequences(encoded_text,
                         maxlen=maxlen,
                         padding='post',
                         truncating='post',
                         value=self.vocab.word2idx['<PAD>'])
