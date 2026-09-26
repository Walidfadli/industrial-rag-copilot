from ingestion import load_documents, split_documents
from embeddings import create_embedding_model, create_embeddings


# 1. Load PDFs
documents = load_documents()

print(f"Pages loaded: {len(documents)}")


# 2. Create chunks
chunks = split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# 3. Extract text from chunks
texts = [chunk.page_content for chunk in chunks]


# 4. Load embedding model
model = create_embedding_model()


# 5. Create embeddings for ALL chunks
embeddings = create_embeddings(model, texts)


# 6. Check result
print(f"Embeddings shape: {embeddings.shape}")