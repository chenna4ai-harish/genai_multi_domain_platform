# from core.services.document_service import DocumentService
#
# # Initialize
# service = DocumentService(domain_id="legal_test_config_temp")
#
# # Upload using file object (now works!)
# with open(r"C:\Users\91917\Desktop\interview_preparation\Project\genai_multi_domain_platform\docs\Sample Leave Policy.pdf", "rb") as f:
#     result = service.upload_document(
#         file_obj=f,
#         metadata={
#             "doc_id": "test_001",
#             "title": "Sample Leave Policy",
#             "domain": "legal_test_config_temp",
#             "doc_type": "policy",
#             "uploader_id": "test"
#         }
#     )
#
# print(f"✅ Uploaded: {result['chunks_ingested']} chunks")
#
# # Query
# results = service.query("vacation policy", top_k=5)
# print(f"✅ Query returned {len(results)} results")
# for r in results[:3]:
#     print(f"  - {r['metadata'].get('doc_id')}: {r['document'][:100]}...")


from core.services.document_service import DocumentService

service = DocumentService(domain_id="legal_test_config_temp")

# 1) Check what retrieval strategies are active
print("Strategies:", list(service.pipeline.retrieval_strategies.keys()))

# 2) List documents seen by the vector store
docs = service.pipeline.list_documents()
print("Docs:", docs)

# 3) Run a query without filters, vector strategy only
results = service.query("vacation", strategy="vector_similarity", top_k=5)
print("Results:", len(results))
for r in results:
    print(r["score"], r["document"][:80])
