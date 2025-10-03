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


# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens


# Function to read documents from a directory
def read_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            documents[filename] = file.read()
    return documents


# Function to read inverted index from a file
def read_inverted_index(file_path):
    inverted_index = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Term: '):
                parts = line.split(', Document IDs: ')
                term = parts[0].replace('Term: ', '').strip()
                doc_ids_str = parts[1].replace('[', '').replace(']', '').strip()
                doc_ids = [doc_id.strip() for doc_id in doc_ids_str.split(', ')]
                inverted_index[term] = doc_ids
    return inverted_index


# Function to calculate TF-IDF for documents with simple logarithmic TF normalization
def calculate_tf_idf(documents, inverted_index):
    N = len(documents)
    tfidf = defaultdict(lambda: defaultdict(float))
    idf = {}

    # Calculate IDF for each term
    for term, postings in inverted_index.items():
        ni = len(set(postings))
        idf[term] = math.log(N / ni) if ni > 0 else 0.0

    # Calculate TF-IDF for each term in each document
    for doc_id, content in documents.items():
        tokens = preprocess_text(content)
        tf = defaultdict(float)
        for term in tokens:
            tf[term] += 1.0  # Simple term frequency

        # Apply simple logarithmic normalization
        tf = {term: math.log(1 + count) for term, count in tf.items()}

        # Calculate TF-IDF values
        for term in tokens:
            tfidf[doc_id][term] = tf.get(term, 0.0) * idf.get(term, 0.0)

    return tfidf, idf


# Function to execute a query and retrieve top results
def execute_query(query, documents, inverted_index, tfidf, idf, output_file):
    query_tokens = preprocess_text(query)
    scores = defaultdict(float)
    query_vector = defaultdict(float)

    # Calculate query vector (simple term frequency)
    for term in query_tokens:
        query_vector[term] += 1.0

    # Apply simple logarithmic normalization for query vector
    query_vector = {term: math.log(1 + count) for term, count in query_vector.items()}

    # Calculate TF-IDF for query vector
    for term in query_tokens:
        query_vector[term] = query_vector.get(term, 0.0) * idf.get(term, 0.0)

    # Calculate cosine similarity between query vector and document vectors
    query_norm = sum(query_vector[term] ** 2 for term in query_vector)
    query_norm = math.sqrt(query_norm) if query_norm != 0 else 1  # Normalize query vector

    for doc_id in documents:
        doc_vector = defaultdict(float)
        for term, weight in tfidf[doc_id].items():
            doc_vector[term] = weight

        # Calculate document vector norm
        doc_norm = sum(doc_vector[term] ** 2 for term in doc_vector)
        doc_norm = math.sqrt(doc_norm) if doc_norm != 0 else 1  # Normalize document vector

        # Calculate dot product
        dot_product = sum(query_vector[term] * doc_vector[term] for term in query_tokens if term in doc_vector)

        # Calculate cosine similarity
        cosine_similarity = dot_product / (query_norm * doc_norm) if query_norm * doc_norm != 0 else 0
        scores[doc_id] = cosine_similarity

    # Sort scores in descending order
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"Top results for query '{query}':\n")
        for rank, (doc_id, score) in enumerate(sorted_scores[:5], start=1):
            file.write(f"{rank}. Document ID: {doc_id}, Score: {score:.4f}\n")

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    # Directory containing documents
    documents_directory = 'collection/docs'

    # Read documents
    documents = read_documents(documents_directory)

    # File path for inverted index
    inverted_index_file = 'inverted_index.py'

    # Read inverted index from file
    inverted_index = read_inverted_index(inverted_index_file)

    # Calculate TF-IDF for documents using the provided inverted index
    tfidf, idf = calculate_tf_idf(documents, inverted_index)

    # Example query
    query = "Is CF mucus abnormal"

    # Output file for results
    output_file = 'query_results.txt'

    # Execute the query and write top results to the output file
    execute_query(query, documents, inverted_index, tfidf, idf, output_file)
