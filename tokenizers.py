from collections import defaultdict
import re
import jsonlines
from typing import List
from tqdm import tqdm
import pickle
import os
import random

# compiled regex for splitting text to words and punctuation
identify_words_regex = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def convert_text_to_words(text: str) -> List[str]:
    """
    Utility function to split text to words and punctuation.
    """
    return identify_words_regex.findall(text)


class Tokenizer(object):
    """
    A generic class that tokenizes text using a given vocabulary.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size  # the maximum number of tokens in the vocabulary
        self.token_to_id = {}  # map from a token (e.g. 'the') to a unique integer id (e.g. 1)
        self.id_to_token = {}  # map from a unique integer id to a token (e.g. 1 -> 'the')

    def tokenize(self, text: str, return_token_ids: bool = False):
        raise Exception("tokenize not implemented")

    def train(self, corpus: List[str]):
        raise Exception("train not implemented")

    def __len__(self):
        raise Exception("__len__ not implemented")


class ReturnWordsTokenizer(Tokenizer):
    def tokenize(self, text: str, return_token_ids: bool = False) -> List[str]:
        return convert_text_to_words(text)

    def train(self, corpus: List[str]):
        pass


class NgramTokenizer(Tokenizer):
    def __init__(self, n: int = 2, vocab_size: int = -1, *args, **kwargs):
        super().__init__(vocab_size, *args, **kwargs)
        self.n = n  # n-gram size

    def tokenize(self, text: str, return_token_ids: bool = False) -> List[List[str]] | List[int]:
        words = convert_text_to_words(text) #splitting words into words and puncation using the utility function
        tokens = [] # returning a list of list token strings

        for i in range(len(words) - self.n + 1): #iterate over the list of words
            ngram = tuple(words[i:i + self.n]) #tuple of each pair of word at a certain index of the list of words
        
            if return_token_ids: #if true 
                if ngram in self.token_to_id: # check if the pair exists in the dictionary 
                    return tokens.append(self.token_to_id[ngram]) #add the token id number
                else:
                    tokens.append(ngram) #if not just add the word pair to the tokens
        return tokens


    def train(self, corpus: List[str]):
        token_counts = defaultdict(int)
        """
        TODO: Train the NgramTokenizer on a corpus. Iterate over the corpus, get ngrams, and count their frequencies.
        Detailed instructions:
        1. Initialize the token counts dictionary.
        2. Iterate over the corpus:
            a. Split the text into words and punctuation using convert_text_to_words utility function.
            b. Get ngrams of size self.n, and count their frequencies.
        3. Limit the vocabulary size to the most frequent self.vocab_size words if self.vocab_size is not -1.
        (vocab_size is the maximum number of tokens in the vocabulary, and -1 means no limit)
        
        Example:
        Input corpus: ["This movie was good", "This movie was bad"]
        Input self.n: 1
        Input self.vocab_size: -1
        Output:
        set self.token_to_id: {"This": 0, "movie": 1, "was": 2, "good": 3, "bad": 4}
        set self.id_to_token: {0: "This", 1: "movie", 2: "was", 3: "good", 4: "bad"}
        """
        raise Exception("TODO: Implement this method")

    def __len__(self):
        return len(self.token_to_id)


if __name__ == "__main__":
    with jsonlines.open("data/imdb_train.txt", "r") as reader:
        dataset = list(reader)

    dataset = dataset[:500]

    corpus = [datapoint["text"] for datapoint in dataset]

    unigram = NgramTokenizer(n=1)
    unigram.train(corpus)

    ngram = NgramTokenizer(n=2)
    ngram.train(corpus)

    sample_text = "I love scifi and am willing to put up with a lot. Scifi movies and TV are usually underfunded, under-appreciated and misunderstood."

    print("Unigram tokens:", unigram.tokenize(sample_text, return_token_ids=False))
    print("-" * 100)
    print("Bigram tokens:", ngram.tokenize(sample_text, return_token_ids=False))
    print("-" * 100)
    print("Unigram token IDs:", unigram.tokenize(sample_text, return_token_ids=True))
    print("-" * 100)
    print("Bigram token IDs:", ngram.tokenize(sample_text, return_token_ids=True))
