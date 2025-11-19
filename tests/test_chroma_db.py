import chromadb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test persistence
persist_dir = Path(".data/test_chromadb")
persist_dir.mkdir(parents=True, exist_ok=True)

try:
    # Try to create client
    logger.info(f"ChromaDB version: {chromadb.__version__}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    logger.info("✅ PersistentClient created successfully")

    # Create a test collection
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"✅ Collection created: {collection.name}")

    # Add a test vector
    collection.add(
        ids=["test1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["test document"],
        metadatas=[{"source": "test"}]
    )
    logger.info("✅ Test vector added")

    # Query it back
    results = collection.query(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=1
    )
    logger.info(f"✅ Query successful: {results}")

    print("\n🎉 ChromaDB is working correctly!")

except Exception as e:
    logger.error(f"❌ Test failed: {e}", exc_info=True)
    print(f"\n❌ Error: {e}")
