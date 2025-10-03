import os
import re
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

# Ensure you have the necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens


def build_inverted_index(documents):
    inverted_index = defaultdict(list)
    term_frequencies = defaultdict(lambda: defaultdict(int))

    for doc_id, content in documents.items():
        tokens = preprocess_text(content)
        for token in tokens:
            inverted_index[token].append(doc_id)
            term_frequencies[doc_id][token] += 1

    return inverted_index, term_frequencies


def compute_idf(inverted_index, total_documents):
    idf = {}
    for term, doc_ids in inverted_index.items():
        idf[term] = math.log(total_documents / len(doc_ids)) + 1  # Adding 1 to avoid division by zero
    return idf


def compute_tf_idf(term_frequencies, idf):
    tf_idf = defaultdict(lambda: defaultdict(float))
    for doc_id, terms in term_frequencies.items():
        for term, freq in terms.items():
            tf = 1 + math.log(freq) if freq > 0 else 0
            tf_idf[doc_id][term] = tf * idf[term]
    return tf_idf


def read_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            documents[filename] = file.read()
    return documents


# Usage
document_directory = 'Collection/docs'
documents = read_documents(document_directory)
inverted_index, term_frequencies = build_inverted_index(documents)

# Compute IDF
total_documents = len(documents)
idf = compute_idf(inverted_index, total_documents)

# Compute TF-IDF
tf_idf = compute_tf_idf(term_frequencies, idf)

# Debugging: Print some of the TF-IDF entries
for doc_id, terms in list(tf_idf.items())[:5]:  # Print the first 5 documents and their TF-IDF scores
    print(f"Document: {doc_id}")
    for term, score in terms.items():
        print(f"  Term: {term}, TF-IDF: {score}")

# Optionally, you can save the TF-IDF scores to a file to inspect them manually
with open('tf_idf_scores.txt', 'w', encoding='utf-8') as f:
    for doc_id, terms in tf_idf.items():
        f.write(f"Document: {doc_id}\n")
        for term, score in terms.items():
            f.write(f"  Term: {term}, TF-IDF: {score}\n")
