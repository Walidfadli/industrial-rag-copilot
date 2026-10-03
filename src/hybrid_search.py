from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "industrial_maintenance"
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_PATH = "qdrant_storage"

BM25_TOP_K = 10
DENSE_TOP_K = 10
HYBRID_TOP_K = 10

# RRF constant
RRF_K = 60


def create_qdrant_client():
    return QdrantClient(path=QDRANT_PATH)


def create_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def load_documents_from_qdrant(client):
    """
    Load all chunks from Qdrant.

    The stored chunks are used to build the BM25 index.
    """

    records = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        records.extend(points)

        if offset is None:
            break

    return records


def tokenize(text):
    """
    Simple tokenizer for BM25.
    """

    return text.lower().split()


def create_bm25_index(records):
    """
    Create a BM25 index from the document chunks.
    """

    documents = [
        record.payload.get("text", "")
        for record in records
    ]

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    return BM25Okapi(tokenized_documents)


def bm25_search(query, records, bm25, top_k=10):
    """
    Keyword-based BM25 search.
    """

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            "record": records[index],
            "bm25_score": float(scores[index])
        })

    return results


def qdrant_search(query, client, model, top_k=10):
    """
    Dense semantic search using Qdrant.
    """

    query_embedding = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    ).points

    return results


def get_record_id(record):
    """
    Create a stable identifier for a chunk.

    chunk_id is preferred because it was created
    during ingestion.
    """

    payload = record.payload

    return payload.get(
        "chunk_id",
        f"{payload.get('source_file')}__"
        f"{payload.get('page')}__"
        f"{payload.get('chunk_index')}"
    )


def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
    top_k=10,
    rrf_k=60
):
    """
    Combine BM25 and dense rankings using
    Reciprocal Rank Fusion (RRF).

    RRF does not combine raw BM25 and cosine
    similarity scores because they use different scales.

    Instead, it combines their rankings.
    """

    fused_scores = {}
    records = {}

    # --------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------

    for rank, item in enumerate(bm25_results, start=1):

        record = item["record"]
        record_id = get_record_id(record)

        records[record_id] = record

        fused_scores.setdefault(record_id, 0.0)

        fused_scores[record_id] += (
            1 / (rrf_k + rank)
        )

    # --------------------------------------------------
    # Dense ranking
    # --------------------------------------------------

    for rank, record in enumerate(dense_results, start=1):

        record_id = get_record_id(record)

        records[record_id] = record

        fused_scores.setdefault(record_id, 0.0)

        fused_scores[record_id] += (
            1 / (rrf_k + rank)
        )

    # --------------------------------------------------
    # Sort by fused RRF score
    # --------------------------------------------------

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    results = []

    for rank, record_id in enumerate(
        ranked_ids[:top_k],
        start=1
    ):

        results.append({
            "record": records[record_id],
            "rrf_score": fused_scores[record_id],
            "rank": rank
        })

    return results


def display_hybrid_results(query, results):

    print("\n" + "=" * 70)
    print("HYBRID SEARCH — RRF")
    print("=" * 70)

    print(f"\nQuery: {query}\n")

    for item in results:

        record = item["record"]
        payload = record.payload

        print(
            f"{item['rank']}. "
            f"{payload.get('source_file')} "
            f"(page {payload.get('page')}) "
            f"→ RRF score {item['rrf_score']:.6f}"
        )

        print(
            f"   Section: {payload.get('section')}"
        )


def main():

    query = "Why can a pump have excessive discharge pressure?"

    print("=" * 70)
    print("INDUSTRIAL MAINTENANCE HYBRID SEARCH")
    print("=" * 70)

    # --------------------------------------------------
    # Initialize
    # --------------------------------------------------

    client = create_qdrant_client()
    model = create_embedding_model()

    # --------------------------------------------------
    # Load documents
    # --------------------------------------------------

    print("\nLoading documents from Qdrant...")

    records = load_documents_from_qdrant(client)

    print(
        f"Documents loaded: {len(records)}"
    )

    # --------------------------------------------------
    # Build BM25
    # --------------------------------------------------

    print("\nBuilding BM25 index...")

    bm25 = create_bm25_index(records)

    print("BM25 index ready.")

    # --------------------------------------------------
    # BM25 search
    # --------------------------------------------------

    print("\nRunning BM25 search...")

    bm25_results = bm25_search(
        query,
        records,
        bm25,
        top_k=BM25_TOP_K
    )

    # --------------------------------------------------
    # Dense search
    # --------------------------------------------------

    print("Running Qdrant dense search...")

    dense_results = qdrant_search(
        query,
        client,
        model,
        top_k=DENSE_TOP_K
    )

    # --------------------------------------------------
    # Hybrid fusion
    # --------------------------------------------------

    print("Combining BM25 + dense results using RRF...")

    hybrid_results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=HYBRID_TOP_K,
        rrf_k=RRF_K
    )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    display_hybrid_results(
        query,
        hybrid_results
    )


if __name__ == "__main__":
    main()