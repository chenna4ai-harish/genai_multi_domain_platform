class ComponentRegistry:
    @staticmethod
    def get_vectorstore_providers():
        return ["chromadb", "pinecone", "faiss", "qdrant"]
    @staticmethod
    def get_chunking_strategies():
        return ["recursive", "semantic"]
    @staticmethod
    def get_embedding_providers():
        return {
            "sentencetransformers": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            "openai": ["text-embedding-ada-002"],
            "gemini": ["models/embedding-001"]
        }

    @staticmethod
    def get_llm_providers():
        # provider -> list of model names
        return {
            "openai": [
                "gpt-4o",
                "gpt-4.1-mini",
                "gpt-3.5-turbo",
            ],
            "gemini": [
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
            ],
            # "ollama": [
            #     "llama3",
            #     "mistral",
            # ],
        }

    @staticmethod
    def get_retrieval_strategies():
        return ["hybrid", "vectorsimilarity", "bm25"]
    @staticmethod
    def get_device_options():
        return ["cpu", "cuda", "mps"]
    @staticmethod
    def get_distance_metrics():
        return ["cosine", "euclidean", "dot_product"]

