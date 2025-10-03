from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

# Function to read documents from a directory
def read_documents_sklearn(directory):
    documents = []
    filenames = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            documents.append(file.read())
            filenames.append(filename)
    return documents, filenames

# Directory containing documents
documents_directory = 'Collection/docs'

# Read documents
documents_sklearn, filenames = read_documents_sklearn(documents_directory)

# Example query
query = "Is CF mucus abnormal"

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)

# Fit and transform documents
tfidf_matrix = vectorizer.fit_transform(documents_sklearn)

# Transform query
query_vector = vectorizer.transform([query])

# Calculate cosine similarity between query and documents
cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Get indices of top 10 scores
top_indices = np.argsort(cosine_similarities)[::-1][:10]

# Print top 10 results
print("Top 10 results for query '{}':".format(query))
for rank, idx in enumerate(top_indices, start=1):
    print(f"{rank}. Document ID: {filenames[idx]}, Score: {cosine_similarities[idx]:.4f}")
