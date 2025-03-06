# ngram-language-model-sentiment-classification

This project implements two core Natural Language Processing tasks:

1. **N-gram Language Modeling:**  
   - Build an N-gram language model (unigrams, bigrams, trigrams, etc.) using an unsupervised subset of the IMDB reviews dataset.
   - Apply text preprocessing (tokenization, lowercasing, punctuation removal) and Laplace smoothing.
   - Evaluate the model using perplexity on a training/testing split.

2. **Sentiment Classification:**  
   - Classify text into five sentiment categories using the Stanford Sentiment Treebank (SST) dataset.
   - Implement two approaches:
     - A Naïve Bayes classifier built from scratch with NumPy and compared with scikit-learn's MultinomialNB.
     - A Logistic Regression model built from scratch (using mini-batch SGD) and compared with scikit-learn's LogisticRegression and SGDClassifier.
   - Generate a confusion matrix and compute precision, recall, and F1 scores from scratch.



