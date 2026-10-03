from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# =========================
# Configuration
# =========================

COLLECTION_NAME = "industrial_maintenance"
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_PATH = "qdrant_storage"


# =========================
# Initialization
# =========================

def create_qdrant_client():
    """
    Connect to the local Qdrant vector database.
    """
    return QdrantClient(path=QDRANT_PATH)


def create_embedding_model():
    """
    Load the same embedding model used during indexing.

    IMPORTANT:
    The query must be embedded using the same model
    that was used to embed the documents.
    """
    return SentenceTransformer(MODEL_NAME)


# =========================
# Retrieval
# =========================

def search(query, client, model, top_k=5):
    """
    Search Qdrant for the chunks most similar to the query.

    Args:
        query: User's question.
        client: Qdrant client.
        model: Sentence Transformer model.
        top_k: Number of chunks to retrieve.

    Returns:
        List of Qdrant search results.
    """

    # Convert the user's question into an embedding
    query_embedding = model.encode(query).tolist()

    # Search the vector database
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    ).points

    return results


# =========================
# Display Results
# =========================

def display_results(query, results):
    """
    Display retrieved chunks in a readable format.
    """

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)

    print(query)

    print("\n" + "=" * 70)
    print(f"TOP {len(results)} RETRIEVED CHUNKS")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")

        print(f"Similarity score : {result.score}")

        print(
            f"Source           : "
            f"{result.payload.get('source_file', 'Unknown')}"
        )

        print(
            f"Page             : "
            f"{result.payload.get('page', 'Unknown')}"
        )

        print("\nText:")

        text = result.payload.get("text", "")

        print(text[:1000])

        print("-" * 70)


# =========================
# Main
# =========================

if __name__ == "__main__":

    # Create Qdrant connection
    client = create_qdrant_client()

    # Load embedding model
    model = create_embedding_model()

    # Test question
    query = "Why can a pump have excessive discharge pressure?"

    # Retrieve relevant chunks
    results = search(
        query=query,
        client=client,
        model=model,
        top_k=5
    )

    # Display results
    display_results(query, results)

    # Close Qdrant connection
    client.close()