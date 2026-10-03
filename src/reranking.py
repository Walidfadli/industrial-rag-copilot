from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def create_reranker():
    """
    Load the Cross-Encoder reranking model.
    """
    return CrossEncoder(MODEL_NAME)


def rerank(query, results, reranker, top_k=3):
    """
    Rerank retrieved documents according to
    their relevance to the query.
    """

    pairs = []

    for result in results:
        text = result.payload.get("text", "")
        pairs.append((query, text))

    scores = reranker.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):
        reranked_results.append(
            {
                "result": result,
                "rerank_score": float(score)
            }
        )

    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]