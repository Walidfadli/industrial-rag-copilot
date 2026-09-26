from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ingestion import load_documents, split_documents
from embeddings import create_embedding_model, create_embeddings


COLLECTION_NAME = "industrial_maintenance"
VECTOR_SIZE = 384


def create_qdrant_client():
    return QdrantClient(path="qdrant_storage")


def create_collection(client):
    collections = client.get_collections().collections

    if COLLECTION_NAME not in [collection.name for collection in collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        print(f"Collection '{COLLECTION_NAME}' created.")

    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")


def insert_documents(client, chunks, embeddings):
    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        point = PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": chunk.page_content,
                "source_file": chunk.metadata.get("source_file"),
                "page": chunk.metadata.get("page"),
            }
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"{len(points)} chunks inserted into Qdrant.")


if __name__ == "__main__":

    # 1. Load documents
    documents = load_documents()
    print(f"Pages loaded: {len(documents)}")

    # 2. Split documents into chunks
    chunks = split_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    # 3. Load embedding model
    model = create_embedding_model()

    # 4. Extract text
    texts = [chunk.page_content for chunk in chunks]

    # 5. Create embeddings
    embeddings = create_embeddings(model, texts)

    print(f"Embeddings shape: {embeddings.shape}")

    # 6. Create Qdrant client
    client = create_qdrant_client()

    # 7. Create collection
    create_collection(client)

    # 8. Insert chunks + embeddings
    insert_documents(client, chunks, embeddings)

    print("Qdrant is ready!")