import gradio as gr
from pathlib import Path
from datetime import datetime
import logging
import yaml
import os

# Import your core pipeline and config manager modules here
from core.pipeline.document_pipeline import DocumentPipeline
from core.config_manager import ConfigManager
from utils.file_parsers.pdf_processor import PDFProcessor
from utils.file_parsers.docx_processor import DOCXProcessor
from utils.file_parsers.txt_processor import TXTProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MultiDomainRAGApp")


class AppState:
    def __init__(self):
        self.config_manager = ConfigManager(config_dir="configs")

        # Get all domain names (returns list of strings like ['hr', 'finance'])
        self.available_domains = self.config_manager.get_all_domain_names()

        # Build domain configs dictionary by loading each domain
        self.domain_confs = {}
        for domain_name in self.available_domains:
            try:
                domain_conf = self.config_manager.load_domain_config(domain_name)
                self.domain_confs[domain_name] = domain_conf
            except Exception as e:
                logger.error(f"Failed to load config for domain {domain_name}: {e}")

        logger.info(f"Initialized AppState with domains: {self.available_domains}")

        # {domain_id: DocumentPipeline instance}
        self.pipeline_cache = {}

        # {domain_id: [document metadata]}
        self.doc_registry = {dom: [] for dom in self.available_domains}

        # Initialize documents for each domain
        self.refresh_documents()

    def get_pipeline(self, domain_id):
        """Get or create pipeline for a domain"""
        if domain_id not in self.pipeline_cache:
            # Pass domain_id STRING, not the config object
            # DocumentPipeline will load the config internally
            self.pipeline_cache[domain_id] = DocumentPipeline(
                domain=domain_id,  # Pass the domain name string
                config_dir="configs"  # Pass config directory
            )
            logger.info(f"Created pipeline for domain: {domain_id}")
        return self.pipeline_cache[domain_id]

    def refresh_domains(self):
        """Reload all domain configurations"""
        self.available_domains = self.config_manager.get_all_domain_names()
        self.domain_confs = {}
        for domain_name in self.available_domains:
            try:
                domain_conf = self.config_manager.load_domain_config(domain_name)
                self.domain_confs[domain_name] = domain_conf
            except Exception as e:
                logger.error(f"Failed to load config for domain {domain_name}: {e}")
        logger.info(f"Refreshed domains: {self.available_domains}")

    def refresh_documents(self):
        """Refresh document list for all domains"""
        for dom in self.available_domains:
            try:
                pipe = self.get_pipeline(dom)
                # Get documents from vector store
                if hasattr(pipe.vectorstore, 'list_documents'):
                    self.doc_registry[dom] = pipe.vectorstore.list_documents()
                else:
                    # Fallback: empty list
                    self.doc_registry[dom] = []
            except Exception as e:
                logger.warning(f"Could not refresh documents for domain {dom}: {e}")
                self.doc_registry[dom] = []

    def add_document_record(self, domain, record):
        """Add a document record to registry"""
        self.doc_registry.setdefault(domain, []).append(record)

    def remove_document_record(self, domain, doc_id):
        """Remove a document record from registry"""
        docs = self.doc_registry.get(domain, [])
        self.doc_registry[domain] = [d for d in docs if d.get('doc_id') != doc_id]

    def get_documents(self, domain):
        """Get all documents for a domain"""
        return self.doc_registry.get(domain, [])


# Initialize app state
app_state = AppState()


# ====== UI Functions and Event Handlers ======
def on_domain_select(domain_id):
    """Handler for domain selection"""
    if not domain_id:
        return "Please select a domain"

    try:
        conf = app_state.domain_confs[domain_id]
        documents = app_state.get_documents(domain_id)

        # Build document summary
        doc_summary = "\n".join([
            f"- **{doc['doc_id']}**: {doc.get('filename', 'Unknown')} ({doc.get('chunk_count', '-')} chunks)"
            for doc in documents
        ]) or "No documents uploaded yet."

        # Extract config details safely using Pydantic model attributes
        return f"""### Domain: {conf.display_name}

**Description:** {conf.description}

**Configuration:**
- **Embedding Provider:** {conf.embeddings.provider}
- **Embedding Model:** {conf.embeddings.model_name}
- **Vector Store:** {conf.vectorstore.provider}
- **Chunking Strategy:** {conf.chunking.strategy}
- **Chunk Size:** {conf.chunking.recursive.chunk_size if conf.chunking.strategy == 'recursive' else 'N/A'}
- **Retrieval Strategy:** {conf.retrieval.strategy}
- **Top-K:** {conf.retrieval.top_k}

**Uploaded Documents:**
{doc_summary}"""
    except Exception as e:
        logger.error(f"Error in on_domain_select: {e}")
        return f"Error loading domain info: {e}"


def on_document_upload(domain_id, file, uploader_id):
    """Handler for document upload"""
    try:
        if not file:
            return "❌ No file provided.", []
        if not domain_id:
            return "❌ Please select a domain.", []

        logger.info(f"Uploading {file.name} to domain {domain_id}")

        # Get pipeline
        pipeline = app_state.get_pipeline(domain_id)
        doc_id = f"{Path(file.name).stem}_{int(datetime.now().timestamp())}"

        # Process document using the pipeline
        result = pipeline.process_document(
            file_path=file.name,
            doc_id=doc_id,
            uploader_id=uploader_id or "anonymous"
        )

        if not result.get('success'):
            return f"❌ Processing failed: {result.get('error', 'Unknown error')}", []

        # Add to registry
        app_state.add_document_record(domain_id, {
            "doc_id": doc_id,
            "filename": file.name,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_count": result.get('chunks_created', 0),  # CORRECT KEY
            "file_size": result.get('file_size', 0)
        })

        # Format metrics for display - READ FROM CORRECT KEYS
        metrics_table = [
            ["Document ID", doc_id],
            ["Chunks Created", result.get('chunks_created', 0)],  # CORRECT KEY
            ["Processing Time", f"{result.get('processing_time', 0):.2f}s"],
            ["Embedding Model", result.get('embedding_model', 'N/A')],
            ["Chunking Strategy", result.get('chunking_strategy', 'N/A')],
            ["Vector Store", result.get('vectorstore', 'N/A')],
            ["File Size", f"{result.get('file_size', 0) / 1024:.1f} KB"],
            ["Status", "✅ Success"]
        ]

        success_msg = f"""✅ Successfully uploaded `{file.name}` as `{doc_id}`

**Processing Summary:**
- **Chunks:** {result.get('chunks_created', 0)}
- **Time:** {result.get('processing_time', 0):.2f}s
- **Model:** {result.get('embedding_model', 'N/A')}
"""

        return success_msg, metrics_table

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", []


def on_query(domain_id, query, top_k):
    """Handler for search queries"""
    try:
        if not query:
            return []
        if not domain_id:
            return []

        logger.info(f"Searching domain '{domain_id}' for: {query}")

        pipeline = app_state.get_pipeline(domain_id)

        # STEP 1: Generate query embedding
        query_embedding = pipeline.embedder.embed_texts([query])[0]

        # STEP 2: Search vector store directly (pipeline has no search method)
        results = pipeline.vectorstore.search(
            query_embedding=query_embedding,
            top_k=int(top_k),
            filters=None  # Optional: add domain filter
        )

        # STEP 3: Format results for display
        res_table = []
        for r in results:
            res_table.append([
                f"{1 - r.get('distance', 0):.4f}",  # Convert distance to score (1 - distance)
                r.get('document', '')[:150] + "..." if len(r.get('document', '')) > 150 else r.get('document', ''),
                r.get('metadata', {}).get('doc_id', ''),
                r.get('id', ''),  # chunk_id
                f"Page: {r.get('metadata', {}).get('page_num', 'N/A')}"
            ])

        logger.info(f"Found {len(results)} results")
        return res_table

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return []



def on_create_domain(domain_id, display_name, desc):
    """Handler for creating new domain"""
    try:
        if not domain_id:
            return "❌ Domain ID is required"
        if domain_id in app_state.available_domains:
            return f"❌ Domain ID `{domain_id}` already exists"

        # Create minimal config
        config = {
            "name": domain_id,
            "display_name": display_name or domain_id,
            "description": desc or f"Domain for {domain_id}",
            "embeddings": {
                "provider": "sentence_transformers",
                "model_name": "all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32,
                "normalize_embeddings": True
            },
            "chunking": {
                "strategy": "recursive",
                "recursive": {
                    "chunk_size": 500,
                    "overlap": 50
                }
            },
            "vectorstore": {
                "provider": "chromadb",
                "chromadb": {
                    "persist_directory": ".data/chromadb",
                    "collection_name": f"{domain_id}_collection"
                }
            },
            "retrieval": {
                "strategy": "hybrid",
                "alpha": 0.7,
                "top_k": 10,
                "enable_metadata_filtering": True,
                "normalize_scores": True
            },
            "security": {
                "allowed_file_types": [".pdf", ".docx", ".txt"],
                "max_file_size_mb": 50
            }
        }

        # Save config
        config_dir = Path("configs/domains")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{domain_id}_domain.yaml"

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        app_state.refresh_domains()
        logger.info(f"Created new domain: {domain_id}")

        # Return updated domain list for dropdowns
        return f"✅ Domain `{display_name or domain_id}` created successfully!"

    except Exception as e:
        logger.error(f"Domain creation error: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"


def on_delete_domain(domain_id, confirm):
    """Handler for deleting domain"""
    try:
        if not confirm:
            return "❌ Please check the confirmation box"
        if not domain_id:
            return "❌ Please select a domain"

        config_path = Path(f"configs/domains/{domain_id}_domain.yaml")
        if config_path.exists():
            config_path.unlink()

        # Clear from cache
        if domain_id in app_state.pipeline_cache:
            del app_state.pipeline_cache[domain_id]

        app_state.refresh_domains()
        logger.info(f"Deleted domain: {domain_id}")
        return f"✅ Domain `{domain_id}` deleted! (Vector store data may still exist)"

    except Exception as e:
        logger.error(f"Domain deletion error: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"


def on_delete_document(domain_id, doc_id, confirm):
    """Handler for deleting document"""
    try:
        if not confirm:
            return "❌ Please check the confirmation box"
        if not domain_id or not doc_id:
            return "❌ Please provide domain and document ID"

        pipeline = app_state.get_pipeline(domain_id)

        # Delete from vector store
        if hasattr(pipeline, 'delete_document'):
            pipeline.delete_document(doc_id)
        elif hasattr(pipeline.vectorstore, 'delete_by_doc_id'):
            pipeline.vectorstore.delete_by_doc_id(doc_id)

        app_state.remove_document_record(domain_id, doc_id)

        logger.info(f"Deleted document {doc_id} from domain {domain_id}")
        return f"✅ Document `{doc_id}` deleted successfully"

    except Exception as e:
        logger.error(f"Document deletion error: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"


def on_chunk_docs_domain_select(domain_id):
    """Get document list for chunk viewer when domain is selected"""
    if not domain_id:
        logger.warning("No domain selected for chunk viewer")
        return gr.Dropdown(choices=[])

    try:
        logger.info(f"Fetching documents for chunk viewer in domain: {domain_id}")

        pipeline = app_state.get_pipeline(domain_id)
        collection = pipeline.vectorstore.collection

        # Get all items from the collection
        all_results = collection.get(include=["metadatas"])

        if not all_results or not all_results.get('metadatas'):
            logger.warning(f"No documents found in domain '{domain_id}'")
            return gr.Dropdown(choices=[])

        # Extract unique doc_ids from metadata
        doc_ids = set()
        for metadata in all_results['metadatas']:
            if metadata and 'doc_id' in metadata:
                doc_ids.add(metadata['doc_id'])

        choices = sorted(list(doc_ids))

        logger.info(f"Found {len(choices)} document(s) in domain '{domain_id}': {choices}")

        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

    except Exception as e:
        logger.error(f"Failed to get documents for domain '{domain_id}': {e}", exc_info=True)
        return gr.Dropdown(choices=[])


def on_chunk_docs_select(domain_id, doc_id):
    """Get chunks for selected document"""
    if not domain_id or not doc_id:
        logger.warning(f"Missing parameters - domain: {domain_id}, doc_id: {doc_id}")
        return []

    try:
        logger.info(f"Fetching chunks for doc_id: '{doc_id}' in domain: '{domain_id}'")

        pipeline = app_state.get_pipeline(domain_id)
        collection = pipeline.vectorstore.collection

        # Get all chunks with matching doc_id using metadata filter
        results = collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"]
        )

        if not results or not results.get('ids'):
            logger.warning(f"No chunks found for doc_id: '{doc_id}'")
            return []

        # Format as table
        table = []
        num_chunks = len(results['ids'])

        for i in range(num_chunks):
            chunk_id = results['ids'][i]
            document = results['documents'][i] if i < len(results['documents']) else ""
            metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}

            table.append([
                chunk_id[:50] if len(chunk_id) > 50 else chunk_id,  # Chunk ID (truncated)
                str(metadata.get('char_start', 'N/A')),  # Start position
                str(metadata.get('char_end', 'N/A')),  # End position
                document[:100] + "..." if len(document) > 100 else document,  # Preview
                str(metadata.get('page_num', 'N/A')),  # Page number
                f"Domain: {metadata.get('domain', 'N/A')}"  # Metadata info
            ])

        logger.info(f"Retrieved {len(table)} chunks for doc_id: '{doc_id}'")
        return table

    except Exception as e:
        logger.error(f"Chunk retrieval error: {e}", exc_info=True)
        return []


# ====== Gradio UI Layout ======

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 GenAI Multi-Domain Knowledge Platform

        **Transform your documents into intelligent, searchable knowledge bases**

        🎯 **Multi-Domain Support** • ⚙️ **Config-Driven** • ** 
        Add new domains (HR, Finance, project related , etc.) with just a YAML file - no code changes needed. 
        Each domain gets its own vector store, chunking strategy, and retrieval configuration.

        **Pipeline:** Upload → Chunk → Embed → Store → Retrieve → Inspect
        ---
        """
    )

    with gr.Tab("1️⃣ Domain Info"):
        gr.Markdown("### Select a domain to view its configuration and uploaded documents")
        domain_dropdown = gr.Dropdown(
            choices=app_state.available_domains,
            label="Select Domain",
            value=app_state.available_domains[0] if app_state.available_domains else None
        )
        info_box = gr.Markdown()

        # Trigger on page load
        demo.load(on_domain_select, inputs=domain_dropdown, outputs=info_box)
        domain_dropdown.change(on_domain_select, inputs=domain_dropdown, outputs=info_box)

    with gr.Tab("2️⃣ Upload Document"):
        gr.Markdown("### Upload documents to be processed and indexed")
        up_domain_dropdown = gr.Dropdown(
            choices=app_state.available_domains,
            label="Select Domain",
            value=app_state.available_domains[0] if app_state.available_domains else None
        )
        file_uploader = gr.File(label="Choose PDF/DOCX/TXT", file_types=[".pdf", ".docx", ".txt"])
        uploader_id = gr.Textbox(label="Uploader ID (optional)", value="")
        upload_btn = gr.Button("🚀 Upload & Process", variant="primary")
        upload_status = gr.Markdown()
        upload_metrics = gr.DataFrame(headers=["Metric", "Value"], label="Processing Metrics")
        upload_btn.click(
            on_document_upload,
            inputs=[up_domain_dropdown, file_uploader, uploader_id],
            outputs=[upload_status, upload_metrics]
        )

    with gr.Tab("3️⃣ Query Documents"):
        gr.Markdown("### Search across your document knowledge base")
        query_domain_dropdown = gr.Dropdown(
            choices=app_state.available_domains,
            label="Select Domain",
            value=app_state.available_domains[0] if app_state.available_domains else None
        )
        query_box = gr.Textbox(label="Enter your question", lines=2, placeholder="What is the vacation policy?")
        query_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Results (Top-K)")
        query_btn = gr.Button("🔍 Search", variant="primary")
        results_table = gr.DataFrame(
            headers=["Score", "Text Snippet", "Doc ID", "Chunk ID", "Metadata"],
            label="Search Results"
        )
        query_btn.click(
            on_query,
            inputs=[query_domain_dropdown, query_box, query_k],
            outputs=results_table
        )

    with gr.Tab("4️⃣ Create Domain"):
        gr.Markdown("### Add a new knowledge domain")
        new_id = gr.Textbox(label="Domain ID (e.g., 'finance')", placeholder="finance")
        new_name = gr.Textbox(label="Display Name", placeholder="Finance Department")
        new_desc = gr.Textbox(label="Description", lines=2, placeholder="Financial policies and procedures")
        new_btn = gr.Button("➕ Create Domain", variant="primary")
        new_msg = gr.Markdown()
        new_btn.click(
            on_create_domain,
            inputs=[new_id, new_name, new_desc],
            outputs=new_msg
        )

    with gr.Tab("5️⃣ Delete Domain"):
        gr.Markdown("### Remove a domain (⚠️ use with caution)")
        del_domain_dropdown = gr.Dropdown(choices=app_state.available_domains, label="Select Domain to Delete")
        confirm_del = gr.Checkbox(label="I understand this will delete the domain configuration")
        del_btn = gr.Button("🗑️ Delete Domain", variant="stop")
        del_status = gr.Markdown()
        del_btn.click(
            on_delete_domain,
            inputs=[del_domain_dropdown, confirm_del],
            outputs=del_status
        )

    with gr.Tab("6️⃣ Manage Documents"):
        gr.Markdown("### Delete specific documents from a domain")
        doc_domain_dropdown = gr.Dropdown(
            choices=app_state.available_domains,
            label="Select Domain",
            value=app_state.available_domains[0] if app_state.available_domains else None
        )
        doc_id_box = gr.Textbox(label="Document ID to Delete", placeholder="employee_handbook_1234567890")
        confirm_doc_del = gr.Checkbox(label="Confirm deletion")
        del_doc_btn = gr.Button("🗑️ Delete Document", variant="stop")
        del_doc_status = gr.Markdown()
        del_doc_btn.click(
            on_delete_document,
            inputs=[doc_domain_dropdown, doc_id_box, confirm_doc_del],
            outputs=del_doc_status
        )

    with gr.Tab("7️⃣ Chunk Viewer"):
        gr.Markdown("### Inspect all chunks for any uploaded document")
        gr.Markdown("**Instructions:** Select a domain, then select a document to view its chunks.")

        chunk_domain_dropdown = gr.Dropdown(
            choices=app_state.available_domains,
            label="Select Domain",
            value=app_state.available_domains[0] if app_state.available_domains else None
        )

        chunk_doc_dropdown = gr.Dropdown(
            choices=[],
            label="Select Document",
            interactive=True
        )

        chunk_refresh_btn = gr.Button("🔄 Refresh Document List", size="sm")

        chunk_table = gr.DataFrame(
            headers=["Chunk ID", "Start", "End", "Preview (100 chars)", "Page", "Metadata"],
            label="Document Chunks",
            wrap=True
        )

        chunk_status = gr.Markdown("")

        # Event handlers
        chunk_domain_dropdown.change(
            on_chunk_docs_domain_select,
            inputs=[chunk_domain_dropdown],
            outputs=[chunk_doc_dropdown]
        )

        chunk_doc_dropdown.change(
            on_chunk_docs_select,
            inputs=[chunk_domain_dropdown, chunk_doc_dropdown],
            outputs=[chunk_table]
        )

        # Refresh button to manually reload document list
        chunk_refresh_btn.click(
            on_chunk_docs_domain_select,
            inputs=[chunk_domain_dropdown],
            outputs=[chunk_doc_dropdown]
        )

    gr.Markdown("---")
    gr.Markdown(
        "💡 **MVP Demo** | Showcasing: Config-Driven Architecture • Hybrid Retrieval • Multi-Domain Support • Chunk-Level Transparency")

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
