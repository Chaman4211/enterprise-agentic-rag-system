import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer

from src.retrieval import semantic_search
from src.vector_store import get_vector_collection

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...", flush=True)
model = SentenceTransformer(MODEL_NAME, device="cpu")

_client, collection = get_vector_collection()

print("Collection:", collection.name)
print("Stored chunks:", collection.count())
print("Distance metric:", collection.metadata.get("hnsw:space"))

query = "How are transformer neural networks used in natural language processing?"

print("\nQuery:", query)
results = semantic_search(query, model, collection, top_k=5)

for index, result in enumerate(results, start=1):
    print(f"\n===== RESULT {index} =====")
    print("ID:", result["id"])
    print("Distance:", round(result["distance"], 4))
    print("Similarity:", round(result["similarity"], 4))
    print("Document:", result["metadata"].get("document_id"))
    print("Text:", result["text"][:700].replace("\n", " "))
