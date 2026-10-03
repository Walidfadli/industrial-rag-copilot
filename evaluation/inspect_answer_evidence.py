import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from retrieval import (
    create_qdrant_client,
    create_embedding_model,
    search,
)

from reranking import (
    create_reranker,
    rerank,
)

TEST_FILE = Path(__file__).parent / "test_questions.json"

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5


def load_test_questions():
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    questions = load_test_questions()

    client = create_qdrant_client()
    embedding_model = create_embedding_model()
    reranker = create_reranker()

    for test in questions:

        question = test["question"]

        print("\n" + "=" * 80)
        print(f"{test['id']}: {question}")
        print("=" * 80)

        retrieved_results = search(
            query=question,
            client=client,
            model=embedding_model,
            top_k=RETRIEVAL_TOP_K,
        )

        reranked_results = rerank(
            query=question,
            results=retrieved_results,
            reranker=reranker,
            top_k=RERANK_TOP_K,
        )

        for rank, item in enumerate(reranked_results, start=1):

            result = item["result"]

            source_file = result.payload.get("source_file")
            page = result.payload.get("page")
            text = result.payload.get("text")

            print("\n" + "-" * 80)
            print(f"RANK {rank}")
            print(f"SOURCE: {source_file}")
            print(f"PAGE: {page}")
            print("-" * 80)
            print(text)

    client.close()


if __name__ == "__main__":
    main()