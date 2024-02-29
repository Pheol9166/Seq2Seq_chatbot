from scipy.sparse import spmatrix
from utils.WordVocab import WordVocab
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextTokenizer:
    def __init__(self, vocab: WordVocab):
        self.vocab = vocab
        self.vectorizer = None

    def text_to_sequence(self, tokenized_text: list[list[str]], sos: str='<SOS>', eos: str='<EOS>') -> list[list[int]]:
        """_integer encode using WordVocab_

        Args:
            tokenized_text (list[list[str]]): _tokenized text_
            sos (str): _start of sentence_
            eos (str): _end of sentence_
        Returns:
            list[list[int]]: _result of integer encoding_
        """

        sequences = []
        for sent in tokenized_text:
            sequence = [sos + self.vocab.word2idx.get(token, self.vocab.word2idx['<UNK>']) + eos for token in sent]
            sequences.append(sequence)
        
        return sequences
    
    def fit_on_texts(self, text: str | list[list[str]], mode: str ='binary') -> None:
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
        if self.vectorizer == None:
            raise ValueError("Tokenizer is None. Call fit_on_texts() first")
        else:
            return self.vectorizer.transform(text)
        
    def padding(self, encoded_text: list[list[int]], maxlen: int=3, sos: bool=False, eos: bool=False) -> list[list[int]]:
        """_Pad the encoded text using Keras's pad_sequences func_

        Args:
            encoded_text (list[list[int]]): _encoded text to be padded_
            maxlen (int, optional): _maximum length of sequences after padding_. Defaults to 3.

        Returns:
            list[list[int]]: _padded sequences_
        """   
        return pad_sequences(encoded_text, maxlen=maxlen, padding='post', truncating='post', value=self.vocab.word2idx['<PAD>'])
