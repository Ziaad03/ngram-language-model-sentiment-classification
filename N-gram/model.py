import os
import re
import string
import random
from collections import defaultdict, Counter
import math
from math import log, exp


def load_imdb_unsup_sentences(folder_path):
    """
    Loads text files from the IMDB 'unsup' (unsupervised) folder.
    split text by newline, strips text, and returns a list of raw lines.
    replace <br /> tags with special token <nl> token.
    """

    all_sentences = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(folder_path, filename), "r", encoding="utf-8"
            ) as file:
                for line in file:
                    line = line.strip().replace("<br />", "<nl>")
                    all_sentences.append(line)

    return all_sentences


def remove_punctuation(text):
    """
    Removes punctuation from the text,
    but keeps <nl> tokens intact.
    """
    text = re.sub(
        r"(?<!<nl>)[{}]+(?!<nl>)".format(re.escape(string.punctuation)), "", text
    )

    return text


def build_vocabulary(sentences):
    """
    lower each sentence,
    Splits each sentence on whitespace, removes punctuation,
    and builds a set of unique tokens (vocabulary).
    """
    vocab = set()

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = remove_punctuation(sentence)
        tokens = sentence.split()
        vocab.update(tokens)

    return vocab


def tokenize(sentences, vocab, unknown="<UNK>"):
    """
    lower each sentence,
    Splits each sentence on whitespace, removes punctuation,
    and replaces tokens not in the vocabulary with unknown token.
    Returns the list of tokenized sentences.
    """
    tokenized_sentences = []

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = remove_punctuation(sentence)
        tokens = sentence.split()
        tokenized_sentence = [token if token in vocab else unknown for token in tokens]
        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences


os.chdir(os.path.dirname(os.path.abspath(__file__)))

imdb_folder = "imdb_data/unsup"
sentences = load_imdb_unsup_sentences(imdb_folder)

print(f"Number of raw sentences loaded: {len(sentences)}")
print(f"Example (first 2 sentences):\n{sentences[:2]}")
