def read_query_relevant_docs(file_path, query_index):
    """Read relevant document IDs for a specific query."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if 0 <= query_index < len(lines):
            relevant_docs = lines[query_index].strip().split()
            # Remove leading zeros from relevant document IDs
            relevant_docs = {doc_id.lstrip('0') for doc_id in relevant_docs}
            return relevant_docs
    return set()

def read_query_results(file_path):
    """Read query results from the file."""
    query_results = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Top results for query") or line.startswith("Document ID"):
                continue
            try:
                parts = line.strip().split(', ')
                if len(parts) == 2:
                    doc_id = parts[0].split(': ')[1].strip().lstrip('0')  # Remove leading zeros
                    score = float(parts[1].split(': ')[1].strip())
                    query_results.append((doc_id, score))
            except ValueError:
                print(f"Skipping line due to ValueError: {line.strip()}")
    print(f"Query Results Read: {query_results}")  # Display results for debugging
    return query_results

def read_colbert_results(file_path):
    """Read ColBERT results from the file."""
    colbert_results = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                doc_id = parts[1].strip().lstrip('0')  # Remove leading zeros
                score = float(parts[3].strip())
                colbert_results.append((doc_id, score))
    return colbert_results

def get_top_n(results, n):
    """Get top N results sorted by score."""
    return sorted(results, key=lambda x: x[1], reverse=True)[:n]

def compute_precision_recall(query_relevant_docs, query_results, top_n=5):
    """Compute Precision@5 and Recall@5 for query results."""
    top_docs = {doc_id for doc_id, _ in get_top_n(query_results, top_n)}
    intersection = query_relevant_docs.intersection(top_docs)

    # Debug prints
    print(f"Top docs: {top_docs}")
    print(f"Intersection: {intersection}")

    # Precision calculation
    precision_at_n = len(intersection) / top_n if top_n > 0 else 0.0

    # Recall calculation
    recall_at_n = len(intersection) / len(query_relevant_docs) if query_relevant_docs else 0.0

    return precision_at_n, recall_at_n

def main():
    query_results_path = 'query_results.txt'
    colbert_results_path = 'experiments/colbert_experiment/main/2024-09/01/03.36.17/colbert_index.ranking.tsv'
    relevant_docs_path = 'collection/Relevant_20'
    query_index = 4  # 5th query (0-based index)

    # Read relevant document IDs for the query
    query_relevant_docs = read_query_relevant_docs(relevant_docs_path, query_index)

    # Debug print
    print(f"Relevant docs: {query_relevant_docs}")

    # Read query results and ColBERT results
    query_results = read_query_results(query_results_path)
    colbert_results = read_colbert_results(colbert_results_path)

    # Compute top N results
    top_query_results = get_top_n(query_results, 5)
    top_colbert_results = get_top_n(colbert_results, 5)

    # Compute Precision@5 and Recall@5 for both results
    precision_query, recall_query = compute_precision_recall(query_relevant_docs, top_query_results)
    precision_colbert, recall_colbert = compute_precision_recall(query_relevant_docs, top_colbert_results)

    # Print results
    print(f"Top 5 query results: {top_query_results}")
    print(f"Top 5 ColBERT results: {[doc_id for doc_id, _ in top_colbert_results]}")
    print(f"Precision@5 for query results: {precision_query:.4f}")
    print(f"Recall@5 for query results: {recall_query:.4f}")
    print(f"Precision@5 for ColBERT results: {precision_colbert:.4f}")
    print(f"Recall@5 for ColBERT results: {recall_colbert:.4f}")

if __name__ == "__main__":
    main()
