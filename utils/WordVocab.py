class WordVocab():
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.idx = 4
        self.count = {}

    def add_word(self, word: str) -> None:
        """_if word not in Vocab, add word in Vocab. else increse its count_

        Args:
            word (str): _word to add_
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.count[word] = 1
            self.idx += 1
        else:
            self.count[word] += 1

    def __len__(self):
        """_return length of Vocab_
        """
        return len(self.word2idx)
    
    def save(self, filename: str):
        """_save Vocab to file_

        Args:
            filename (str): _name of file_
        """
        with open(filename, 'w') as f:
            for (word, idx), count in zip(self.word2idx.items(), self.count):
                f.write(f"{word}\t{idx}\t{count}\n")