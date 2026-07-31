"""
NLP Weekly Task 4: Bag of Words and TF-IDF
--------------------------------------------
This script demonstrates two core text-vectorization techniques in NLP:

1. Bag of Words (BoW)  -> counts how many times each word appears in a document
2. TF-IDF              -> weighs words by importance (frequent in a document,
                           but rare across the whole corpus)

We implement both using scikit-learn's CountVectorizer and TfidfVectorizer,
and also show a small "from scratch" version so the underlying math is clear.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------
# 1. Sample corpus
# ---------------------------------------------------------------
corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are great pets",
    "The dog chased the cat"
]

print("=" * 70)
print("CORPUS")
print("=" * 70)
for i, doc in enumerate(corpus):
    print(f"Doc {i}: {doc}")

# ---------------------------------------------------------------
# 2. Bag of Words using CountVectorizer
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("1. BAG OF WORDS (CountVectorizer)")
print("=" * 70)

bow_vectorizer = CountVectorizer(lowercase=True, stop_words=None)
bow_matrix = bow_vectorizer.fit_transform(corpus)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out(),
    index=[f"Doc {i}" for i in range(len(corpus))]
)
print("\nVocabulary size:", len(bow_vectorizer.vocabulary_))
print("\nBoW Matrix (rows = documents, columns = word counts):\n")
print(bow_df)

# ---------------------------------------------------------------
# 3. TF-IDF using TfidfVectorizer
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("2. TF-IDF (TfidfVectorizer)")
print("=" * 70)

tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out(),
    index=[f"Doc {i}" for i in range(len(corpus))]
).round(3)

print("\nTF-IDF Matrix (rows = documents, columns = tf-idf weights):\n")
print(tfidf_df)

# Show top 3 most important words per document
print("\nTop 3 highest-weighted words per document:")
for i in range(len(corpus)):
    row = tfidf_df.iloc[i]
    top_words = row.sort_values(ascending=False).head(3)
    print(f"  Doc {i}: {list(zip(top_words.index, top_words.values))}")

# ---------------------------------------------------------------
# 4. Bonus: Manual "from scratch" TF-IDF for one word, to show the math
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3. MANUAL TF-IDF CALCULATION (for understanding the math)")
print("=" * 70)


def compute_tf(word, document):
    words = document.lower().split()
    return words.count(word) / len(words)


def compute_idf(word, all_documents):
    n_docs_containing_word = sum(
        1 for doc in all_documents if word in doc.lower().split()
    )
    # +1 smoothing to avoid division by zero (same idea sklearn uses)
    return np.log((1 + len(all_documents)) / (1 + n_docs_containing_word)) + 1


word_to_check = "cat"
print(f"\nManually computing TF-IDF for the word: '{word_to_check}'\n")
for i, doc in enumerate(corpus):
    tf = compute_tf(word_to_check, doc)
    idf = compute_idf(word_to_check, corpus)
    print(f"  Doc {i}: TF={tf:.3f}, IDF={idf:.3f}, TF-IDF={tf * idf:.3f}")

print("\n(Compare these values with the 'cat' column in the TF-IDF table above.")
print(" They match the underlying formula sklearn uses, modulo L2 row normalization.)")

print("\nDone. Both BoW and TF-IDF matrices were successfully generated.")
