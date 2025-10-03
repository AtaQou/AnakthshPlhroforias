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


# Function to calculate TF-IDF for documents with a given TF method
def calculate_tf_idf(documents, inverted_index, tf_method='raw'):
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

        if tf_method == 'logarithmic':
            for term in tf:
                tf[term] = 1 + math.log(tf[term])

        max_tf = max(tf.values(), default=1.0)
        for term in tokens:
            tfidf[doc_id][term] = (tf[term] / max_tf) * idf.get(term, 0.0)

    return tfidf, idf


# Function to execute a query and retrieve top results
def execute_query(query, documents, inverted_index, tfidf, idf, tf_method):
    query_tokens = preprocess_text(query)
    scores = defaultdict(float)
    query_vector = defaultdict(float)

    # Calculate query vector (simple term frequency)
    for term in query_tokens:
        query_vector[term] += 1.0

    if tf_method == 'logarithmic':
        for term in query_vector:
            query_vector[term] = 1 + math.log(query_vector[term])

    # Calculate TF-IDF for query vector
    max_tf_query = max(query_vector.values(), default=1.0)
    for term in query_tokens:
        query_vector[term] = (query_vector[term] / max_tf_query) * idf.get(term, 0.0)

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

    # Print top results
    print(f"Top results for query '{query}' using {tf_method} TF method:")
    for rank, (doc_id, score) in enumerate(sorted_scores[:5], start=1):
        print(f"{rank}. Document ID: {doc_id}, Score: {score:.4f}")

        # Print top TF-IDF values for the document
        sorted_tfidf = sorted(tfidf[doc_id].items(), key=lambda x: x[1], reverse=True)[:5]
        print("   Top TF-IDF values:")
        for term, tfidf_value in sorted_tfidf:
            print(f"   Term: {term}, TF-IDF: {tfidf_value:.4f}, IDF: {idf.get(term, 0.0):.4f}")

        print()  # Print an empty line for separation

    return sorted_scores


if __name__ == "__main__":
    # Directory containing documents
    documents_directory = 'collection/docs'

    # Read documents
    documents = read_documents(documents_directory)

    # File path for inverted index
    inverted_index_file = 'inverted_index.txt'

    # Read inverted index from file
    inverted_index = read_inverted_index(inverted_index_file)

    # Calculate TF-IDF for documents using raw count TF method
    tfidf_raw, idf_raw = calculate_tf_idf(documents, inverted_index, tf_method='raw')

    # Calculate TF-IDF for documents using logarithmic TF method
    tfidf_log, idf_log = calculate_tf_idf(documents, inverted_index, tf_method='logarithmic')

    # Example query
    query = "Is CF mucus abnormal"

    # Execute the query and retrieve top results for raw count TF method
    top_results_raw = execute_query(query, documents, inverted_index, tfidf_raw, idf_raw, tf_method='raw')

    # Execute the query and retrieve top results for logarithmic TF method
    top_results_log = execute_query(query, documents, inverted_index, tfidf_log, idf_log, tf_method='logarithmic')
