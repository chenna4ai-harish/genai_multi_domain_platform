# test_fixed.py
import hashlib
from core.factories.chunking_factory import ChunkingFactory
from core.config_manager import ChunkingConfig, RecursiveChunkingConfig

print("="*70)
print("TESTING FIXED FACTORY")
print("="*70)

# Create config
config = ChunkingConfig(
    strategy="recursive",
    recursive=RecursiveChunkingConfig(chunk_size=500, overlap=50)
)

# Create chunker
print("\n📋 Creating chunker...")
chunker = ChunkingFactory.create_chunker(
    config=config,
    embedding_model_name="all-MiniLM-L6-v2"
)

print(f"✅ Chunker created: {type(chunker).__name__}")
print(f"   chunk_size: {chunker.chunk_size}")
print(f"   overlap: {chunker.overlap}")

# Test chunking WITH valid SHA-256 hash
print("\n🧪 Testing chunking...")
text = "This is a test sentence. " * 50

# Generate a valid SHA-256 hash
file_hash = hashlib.sha256(text.encode()).hexdigest()
print(f"   Generated valid SHA-256 hash: {file_hash[:32]}...")

chunks = chunker.chunk_text(
    text=text,
    doc_id="test_doc_001",
    domain="test_domain",
    source_file_path=r"C:\Users\91917\Desktop\interview_preparation\Project\genai_multi_domain_platform\docs\test.txt",
    file_hash=file_hash  # ← Now a valid 64-character SHA-256 hash
)

print(f"✅ Created {len(chunks)} chunks from {len(text)} chars")

# Show chunk details
if chunks:
    print(f"\n📄 Chunk details:")
    for idx, chunk in enumerate(chunks[:3], 1):  # Show first 3
        if hasattr(chunk, '__dict__'):
            print(f"\n   Chunk {idx}:")
            print(f"     Type: {type(chunk).__name__}")
            if hasattr(chunk, 'text'):
                preview = chunk.text[:80] + "..." if len(chunk.text) > 80 else chunk.text
                print(f"     Text: {preview}")
            if hasattr(chunk, 'doc_id'):
                print(f"     Doc ID: {chunk.doc_id}")
            if hasattr(chunk, 'chunk_index'):
                print(f"     Index: {chunk.chunk_index}")
            if hasattr(chunk, 'char_start') and hasattr(chunk, 'char_end'):
                print(f"     Position: {chunk.char_start}-{chunk.char_end}")
        else:
            preview = str(chunk)[:80] + "..." if len(str(chunk)) > 80 else str(chunk)
            print(f"   Chunk {idx}: {preview}")
else:
    print("⚠️  No chunks created (check logs above for validation errors)")

print("\n" + "="*70)
print("SUCCESS!")
print("="*70)
