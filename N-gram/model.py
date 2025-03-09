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
                    line = line.strip().replace("<br />", " <nl> ")
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


def split_data(sentences, test_split=0.1):
    """
    shuffle the sentences
    split them into train and test sets (first 1-test_split of the data is the training)
    return the train and test sets
    """
    random.shuffle(sentences)
    split_index = int(len(sentences) * (1 - test_split))
    train_sentences = sentences[:split_index]
    test_sentences = sentences[split_index:]

    return train_sentences, test_sentences


def pad_sentence(tokens, n):
    """
    Pads a list of tokens with <s> at the start (n-1 times)
    and </s> at the end (once).
    For example, if n=3, you add 2 <s> tokens at the start.
    """
    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
    return padded


def build_ngram_counts(tokenized_sentences, n):
    """
    Builds n-gram counts and (n-1)-gram counts from the given tokenized sentences.
    Each sentence is padded with <s> and </s>.

    Args:
        tokenized_sentences: list of lists, where each sub-list is a tokenized sentence.
        n: the order of the n-gram (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
        ngram_counts: Counter of n-grams (tuples of length n).
        context_counts: Counter of (n-1)-gram contexts.
    """
    ngram_counts = Counter()
    context_counts = Counter()

    for sentence in tokenized_sentences:
        padded_sentence = pad_sentence(sentence, n)
        for i in range(len(padded_sentence) - n + 1):
            ngram = tuple(padded_sentence[i : i + n])
            context = tuple(padded_sentence[i : i + n - 1])
            ngram_counts[ngram] += 1
            context_counts[context] += 1

    return ngram_counts, context_counts


def laplace_probability(ngram, ngram_counts, context_counts, vocab_size, alpha=1.0):
    """
    Computes the probability of an n-gram using Laplace (add-alpha) smoothing.

    P(w_i | w_{i-(n-1)}, ..., w_{i-1}) =
        (count(ngram) + alpha) / (count(context) + alpha * vocab_size)

    Args:
        ngram: tuple of tokens representing the n-gram
        ngram_counts: Counter of n-grams
        context_counts: Counter of (n-1)-gram contexts
        vocab_size: size of the vocabulary
        alpha: smoothing parameter (1.0 = add-1 smoothing)

    Returns:
        Probability of the given n-gram.
    """
    ngram_count = ngram_counts[ngram]
    context = ngram[:-1]
    context_count = context_counts[context]
    prob = (ngram_count + alpha) / (context_count + alpha * vocab_size)
    return prob


def predict_next_token(
    context_tokens, ngram_counts, context_counts, vocab, n=2, alpha=1.0, top_k=5
):
    """
    Given a list of context tokens, predict the next token using the n-gram model.
    Returns the top_k predictions as (token, probability).
    """
    context = tuple(context_tokens[-(n - 1) :])
    candidates = []

    for token in vocab:
        ngram = context + (token,)
        prob = laplace_probability(
            ngram, ngram_counts, context_counts, len(vocab), alpha
        )
        candidates.append((token, prob))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def generate_text_with_limit(
    start_tokens, ngram_counts, context_counts, vocab, n=2, alpha=1.0, max_length=20
):
    """
    Generates text from an n-gram model until it sees </s>
    or reaches a maximum total length (max_length).

    Args:
        start_tokens (list): initial context to begin generation
        ngram_counts (Counter): trained n-gram counts
        context_counts (Counter): trained (n-1)-gram counts
        vocab (set): the model vocabulary
        n (int): n-gram order, 2 for bigram, 3 for trigram, etc.
        alpha (float): Laplace smoothing parameter
        max_length (int): maximum number of tokens to generate (including start_tokens)

    Returns:
        A list of tokens representing the generated sequence.
    """
    generated = start_tokens[:]

    while len(generated) < max_length:
        context_tokens = generated[-(n - 1) :]
        next_token_candidates = predict_next_token(
            context_tokens,
            ngram_counts,
            context_counts,
            vocab,
            n,
            alpha,
            top_k=10,  # Increased top_k
        )

        if not next_token_candidates:
            break

        # Use weighted random choice based on probabilities
        total_prob = sum(prob for _, prob in next_token_candidates)
        rand_val = random.random() * total_prob
        cumulative = 0

        for token, prob in next_token_candidates:
            cumulative += prob
            if cumulative > rand_val:
                next_token = token
                break
        else:
            next_token = next_token_candidates[0][0]

        if next_token == "</s>":
            break

        generated.append(next_token)

    return generated


def calculate_perplexity(
    tokenized_sentences, ngram_counts, context_counts, vocab_size, n=2, alpha=1.0
):
    """
    Calculates the perplexity of an n-gram model (with Laplace smoothing)
    on a list of tokenized sentences.

    Args:
        tokenized_sentences: List of lists of tokens.
        ngram_counts: Counter of n-grams.
        context_counts: Counter of (n-1)-grams.
        vocab_size: Size of the vocabulary.
        n: n-gram order.
        alpha: Laplace smoothing parameter.

    Returns:
        A float representing the perplexity on the given dataset.
    """
    log_prob_sum = 0.0
    total_tokens = 0

    for sentence in tokenized_sentences:
        padded_sentence = pad_sentence(sentence, n)
        for i in range(len(padded_sentence) - n + 1):
            ngram = tuple(padded_sentence[i : i + n])
            prob = laplace_probability(
                ngram, ngram_counts, context_counts, vocab_size, alpha
            )
            log_prob_sum += log(prob)
            total_tokens += 1

    perplexity = exp(-log_prob_sum / total_tokens)
    return perplexity


def main():
    # Processing Data
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    imdb_folder = "imdb_data/unsup"
    sentences = load_imdb_unsup_sentences(imdb_folder)

    # print(f"Number of raw sentences loaded: {len(sentences)}")
    # print(f"Example (first 2 sentences):\n{sentences[:2]}")

    assert len(sentences) == 50000, "Expected 50,000 sentences from the unsup folder."

    random.seed(42)

    train_sentences, test_sentences = split_data(sentences)

    # print(f"Number of training sentences: {len(train_sentences)}")
    # print(f"Number of test sentences: {len(test_sentences)}")

    assert len(train_sentences) == 45000, "Expected 45,000 sentences for training."
    assert len(test_sentences) == 5000, "Expected 5,000 sentences for testing."

    vocab = build_vocabulary(train_sentences)
    tokenized_sentences = tokenize(train_sentences, vocab)

    # print(f"Vocabulary size: {len(vocab)}")
    # print(
    #     f"Example tokens from first sentence: {tokenized_sentences[0][:10] if tokenized_sentences else 'No tokens loaded'} ..."
    # )

    assert (
        len(tokenized_sentences) == 45000
    ), "Expected tokenized sentences count to match raw sentences."

    example = "I love Natural language processing, and i want to be a great engineer."
    assert (
        len(example) == 70
    ), "Example sentence length (in characters) does not match the expected 70."

    example_tokens = tokenize([example], vocab)[0]
    assert (
        len(example_tokens) == 13
    ), "Token count for the example sentence does not match the expected 13."

    # Building N-gram
    n = 4
    alpha = 2
    ngram_counts, context_counts = build_ngram_counts(tokenized_sentences, n=n)
    # print(f"Number of bigrams: {len(ngram_counts)}")
    # print(f"Number of contexts: {len(context_counts)}")

    context = ["i", "love"]
    generated_seq = generate_text_with_limit(
        start_tokens=context,
        ngram_counts=ngram_counts,
        context_counts=context_counts,
        vocab=vocab,
        n=n,
        alpha=alpha,
        max_length=128,
    )

    print("Generated Sequence:", generated_seq)

    test_tockenized_sentences = tokenize(test_sentences, vocab)
    print(
        f"Preplexity: {calculate_perplexity(test_tockenized_sentences, ngram_counts, context_counts, len(vocab), n, alpha)}"
    )


if __name__ == "__main__":
    main()
