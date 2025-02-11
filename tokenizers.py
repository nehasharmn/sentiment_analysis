from collections import defaultdict
import re
import jsonlines
from typing import List
from tqdm import tqdm
import pickle
import os
import random

#HIIIIIIIIIIIIIIIIIIIIIIIII
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
        splits = convert_text_to_words(text)
        ngrams = []
        for i in range(len(splits) - self.n + 1):
            temp_token = tuple(splits[i:i+self.n])
            if temp_token in self.token_to_id:
                ngrams.append(temp_token)

        if return_token_ids:
            return [self.token_to_id[ngram] for ngram in ngrams]
        
        return ngrams


    def train(self, corpus: List[str]):
        token_freq = defaultdict(int)

        for text in tqdm(corpus):
            words = convert_text_to_words(text)
            for i in range(len(words) - self.n + 1):
                token = tuple(words[i:i+self.n])
                token_freq[token] += 1

        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)

        if self.vocab_size != -1:
            sorted_tokens = sorted_tokens[:self.vocab_size]

        for idx, (token, _) in enumerate(sorted_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

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
