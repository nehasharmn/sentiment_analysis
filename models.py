# models.py

from utils import SentimentExample
from typing import List
from collections import Counter
import time
from tokenizers import Tokenizer
import numpy as np
from tqdm import tqdm
import gensim.downloader as api


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a text and returns an indexed list of features.
    """

    def extract_features(self, text: str) -> Counter:
        """
        Extract features from a text represented as a list of words.
        :param text: words in the example to featurize
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")



class CountFeatureExtractor(FeatureExtractor):
    """
    Extracts count features from text - your tokenizer returns token ids; you count their occurences.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenizer)

    def extract_features(self, text: str) -> Counter:
        tokens = self.tokenizer.tokenize(text, return_token_ids=True)
        return Counter(tokens)


class CustomFeatureExtractor(FeatureExtractor):
    """
    Custom feature extractor combining n-grams and review metadata (e.g., review length).
    - Combines unigram and bigram counts.
    - Adds the total number of tokens and the average token length as features.
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        # Feature count includes token IDs plus additional metadata features (e.g., length)
        return len(self.tokenizer) + 2

    def extract_features(self, text: str) -> Counter:
        token_ids = self.tokenizer.tokenize(text, return_token_ids=True)
        feature_counts = Counter(token_ids)

        total_tokens = len(token_ids)
        avg_token_length = sum(len(self.tokenizer.id_to_token[id]) for id in token_ids) / total_tokens if total_tokens > 0 else 0

        feature_counts[len(self.tokenizer)] = total_tokens
        feature_counts[len(self.tokenizer) + 1] = avg_token_length

        return feature_counts

class MeanPoolingWordVectorFeatureExtractor(FeatureExtractor):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        print("Loading word2vec model...")
        self.word_to_vector_model = api.load("glove-twitter-25")  
        print("Word2vec model loaded")

    def __len__(self):
        return 25

    def get_word_vector(self, word) -> np.ndarray:
        # Handle tuple case from n-gram tokenizer
        if isinstance(word, tuple):
            word = word[0]
        
        try:
            if word in self.word_to_vector_model:
                return self.word_to_vector_model[word]
        except:
            pass
        
        return np.zeros(25)

    def extract_features(self, text: str) -> dict:
        words = self.tokenizer.tokenize(text)
        vectors = []
        
        for word in words:
            if isinstance(word, tuple):
                word = word[0]
            try:
                if word in self.word_to_vector_model:
                    vector = self.word_to_vector_model[word]
                    vectors.append(vector)
            except:
                continue
        
        if not vectors:
            return {i: 0.0 for i in range(25)}
            
        vectors = np.array(vectors)
        mean_vector = np.mean(vectors, axis=0)
        
        return {i: float(mean_vector[i]) for i in range(25)}

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, text: List[str]) -> int:
        """
        :param text: words (List[str]) in the text to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, text: List[str]) -> int:
        return 1


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function, avoids overflow.
    A utility function for the logistic regression classifier.
    """
    if x < 0:
        return np.exp(x) / (1 + np.exp(x))
    return 1 / (1 + np.exp(-x))

class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, featurizer: FeatureExtractor):
        self.featurizer = featurizer
        self.weights = np.zeros(len(self.featurizer), dtype=np.float64)
        self.bias = 0

    def predict(self, text: str) -> int:
        features = self.featurizer.extract_features(text)
        
        # If features is a Counter or dict, convert to vector form
        if isinstance(features, (Counter, dict)):
            feature_vector = np.zeros(len(self.featurizer))
            for idx, value in features.items():
                feature_vector[idx] = value
            features = feature_vector
            
        score = self.bias + np.dot(self.weights, features)
        prob = sigmoid(score)
        return 1 if prob >= 0.5 else 0

    def training_step(self, batch_exs: List[SentimentExample], learning_rate: float):
        gradient_w = np.zeros_like(self.weights)
        gradient_b = 0.0

        for ex in batch_exs:
            features = self.featurizer.extract_features(ex.words)
            
            # Convert features to vector if necessary
            if isinstance(features, (Counter, dict)):
                feature_vector = np.zeros(len(self.featurizer))
                for idx, value in features.items():
                    feature_vector[idx] = value
                features = feature_vector
                
            prediction = self.predict(ex.words)
            error = prediction - ex.label
            
            gradient_w += error * features
            gradient_b += error

        self.weights -= (learning_rate / len(batch_exs)) * gradient_w
        self.bias -= (learning_rate / len(batch_exs)) * gradient_b

        

def get_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    Calculate the accuracy of the predictions.
    """
    num_correct = 0
    num_total = len(predictions)
    for i in range(num_total):
        if predictions[i] == labels[i]:
            num_correct += 1
    return num_correct / num_total


def run_model_over_dataset(
    model: SentimentClassifier, dataset: List[SentimentExample]
) -> List[int]:
    """
    Run the model over a dataset and return the predictions.
    """
    predictions = []
    for ex in dataset:
        predictions.append(model.predict(ex.words))
    return predictions


def train_logistic_regression(
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    feat_extractor: FeatureExtractor,
    learning_rate: float = 0.01,
    batch_size: int = 10,
    epochs: int = 10,
) -> LogisticRegressionClassifier:

    model = LogisticRegressionClassifier(feat_extractor)
    best_dev_accuracy = 0.0
    best_weights = None
    best_bias = None

    schedule = lambda epoch: learning_rate * (0.95**epoch)  # Decays learning rate over time

    pbar = tqdm(range(epochs))  
    for epoch in pbar:
        shuffle_train_exs = np.random.permutation(train_exs) 

    
        for i in range(0, len(shuffle_train_exs), batch_size): 
            batch_exs = shuffle_train_exs[i:i + batch_size] 
            current_learning_rate = schedule(epoch)
            model.training_step(batch_exs, current_learning_rate) #update

       
        predictions = [model.predict(y.words) for y in dev_exs] 
        development_labels = [x.label for x in dev_exs]
        deveoplment_accuracy = get_accuracy(predictions, development_labels)

        if deveoplment_accuracy > best_dev_accuracy: # copys the best model
            best_dev_accuracy = deveoplment_accuracy
            best_weights = model.weights.copy()
            best_bias = model.bias

        metrics = {"best_dev_acc": best_dev_accuracy, "cur_dev_acc": deveoplment_accuracy} #logging current epoch performance using tqdm
        pbar.set_postfix(metrics)

    
    #set weights and model
    model.weights = best_weights
    model.bias = best_bias

    return model


def train_model(
    args,
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    tokenizer: Tokenizer,
    learning_rate: float,
    batch_size: int,
    epochs: int,
) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.feats == "COUNTER":
        feat_extractor = CountFeatureExtractor(tokenizer)
    elif args.feats == "WV":
        feat_extractor = MeanPoolingWordVectorFeatureExtractor(tokenizer)
    elif args.feats == "CUSTOM":
        feat_extractor = CustomFeatureExtractor(tokenizer)

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "LR":
        model = train_logistic_regression(
            train_exs,
            dev_exs,
            feat_extractor,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
        )
    else:
        raise Exception("Pass in TRIVIAL or LR to run the appropriate system")
    return model
