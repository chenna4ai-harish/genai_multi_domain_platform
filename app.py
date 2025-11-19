"""
app.py - Multi-Domain Document Intelligence Platform

New Features:
-------------
1. Document listing per domain
2. Upload confirmation messages
3. Create new domains
4. Delete domains
5. Document management (list, delete)
6. Better PDF/DOCX extraction
"""

import gradio as gr
from pathlib import Path
import time
import logging
import yaml
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime

# Core pipeline
from core.pipeline.document_pipeline import DocumentPipeline
from core.config_manager import ConfigManager

# File processors
from utils.file_parsers.pdf_processor import PDFProcessor
from utils.file_parsers.docx_processor import DOCXProcessor
from utils.file_parsers.txt_processor import TXTProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Global State Management
# =============================================================================

class AppState:
    """Enhanced application state with document tracking."""

    def __init__(self):
        self.config_manager = None
        self.pipelines: Dict[str, DocumentPipeline] = {}
        self.available_domains: List[str] = []
        self.current_domain: Optional[str] = None
        self.uploaded_documents: Dict[str, List[Dict]] = {}  # Track docs per domain

        # Initialize
        self._initialize()

    def _initialize(self):
        """Initialize config manager and discover domains."""
        try:
            # Calculate project root dynamically
            current_file = Path(__file__).resolve()

            # Try multiple locations
            if (current_file.parent / "configs").exists():
                project_root = current_file.parent
            elif (current_file.parent.parent / "configs").exists():
                project_root = current_file.parent.parent
            else:
                project_root = current_file.parent
                while project_root != project_root.parent:
                    if (project_root / "configs").exists():
                        break
                    project_root = project_root.parent

            self.config_dir = project_root / "configs"

            if not self.config_dir.exists():
                raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

            logger.info(f"📁 Config directory: {self.config_dir}")

            # Initialize config manager
            self.config_manager = ConfigManager(config_dir=str(self.config_dir))

            # Get available domains
            self.available_domains = self.config_manager.get_all_domain_names()

            # Initialize document tracking
            for domain in self.available_domains:
                self.uploaded_documents[domain] = []

            logger.info(f"✅ Initialized with {len(self.available_domains)} domains")

        except Exception as e:
            logger.error(f"Failed to initialize AppState: {e}")
            self.available_domains = []

    def get_pipeline(self, domain: str) -> DocumentPipeline:
        """Get or create pipeline for domain (cached)."""
        if domain not in self.pipelines:
            logger.info(f"Creating pipeline for domain: {domain}")
            self.pipelines[domain] = DocumentPipeline(
                domain=domain,
                config_dir=str(self.config_dir)
            )
        return self.pipelines[domain]

    def add_document_record(self, domain: str, doc_info: Dict):
        """Track uploaded document."""
        if domain not in self.uploaded_documents:
            self.uploaded_documents[domain] = []
        self.uploaded_documents[domain].append(doc_info)

    def get_documents(self, domain: str) -> List[Dict]:
        """Get all documents for a domain."""
        return self.uploaded_documents.get(domain, [])

    def remove_document_record(self, domain: str, doc_id: str):
        """Remove document from tracking."""
        if domain in self.uploaded_documents:
            self.uploaded_documents[domain] = [
                doc for doc in self.uploaded_documents[domain]
                if doc['doc_id'] != doc_id
            ]

    def refresh_domains(self):
        """Refresh domain list after create/delete."""
        self.available_domains = self.config_manager.get_all_domain_names()

        # Initialize tracking for new domains
        for domain in self.available_domains:
            if domain not in self.uploaded_documents:
                self.uploaded_documents[domain] = []


# Initialize global state
app_state = AppState()


# =============================================================================
# Helper Functions
# =============================================================================

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def extract_text_from_file(file_path: str) -> tuple[str, str]:
    """
    Extract text using project's file processors.

    This is just a thin wrapper - actual logic is in utils/file_parsers/
    """
    file_path_obj = Path(file_path)
    file_extension = file_path_obj.suffix.lower()

    try:
        if file_extension == '.pdf':
            # Use project's PDF processor
            processor = PDFProcessor(backend="pymupdf")  # or "pypdf2" as fallback
            text = processor.extract_text(str(file_path_obj))
            return text, "PDF"

        elif file_extension == '.docx':
            # Use project's DOCX processor
            processor = DOCXProcessor()
            text = processor.extract_text(str(file_path_obj))
            return text, "DOCX"

        elif file_extension == '.txt':
            # Use project's TXT processor
            processor = TXTProcessor()
            text = processor.extract_text(str(file_path_obj))
            return text, "TXT"

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise RuntimeError(f"Could not extract text: {e}")


# =============================================================================
# UI Functions - Domain Management
# =============================================================================

def on_domain_select(domain_name: str) -> tuple[str, str, str]:
    """Handle domain selection with document count."""
    try:
        app_state.current_domain = domain_name
        pipeline = app_state.get_pipeline(domain_name)
        config = pipeline.config

        # Get document count
        doc_count = len(app_state.get_documents(domain_name))

        info_md = f"""
## 🏢 Domain: {config.display_name}

**Description:** {config.description}  
**Documents Uploaded:** {doc_count}

### ⚙️ Configuration

**Chunking:** `{config.chunking.strategy}` (size: {config.chunking.recursive.chunk_size}, overlap: {config.chunking.recursive.overlap})  
**Embedding:** `{config.embeddings.provider}` ({config.embeddings.model_name})  
**Vector Store:** `{config.vector_store.provider}` ({config.vector_store.chromadb.collection_name if config.vector_store.provider == 'chromadb' else 'N/A'})

---
✅ Ready for document processing!
        """

        status_msg = f"✅ Selected: **{config.display_name}** ({doc_count} documents)"

        # Return document list
        docs_md = format_document_list(domain_name)

        return info_md, status_msg, docs_md

    except Exception as e:
        error_msg = f"❌ Error loading domain: {str(e)}"
        return "Error loading domain", error_msg, ""


def format_document_list(domain: str) -> str:
    """Format document list for display."""
    documents = app_state.get_documents(domain)

    if not documents:
        return "📝 **No documents uploaded yet**\n\nUpload documents in Tab 2 to see them here."

    md = f"## 📚 Documents in {domain.upper()} Domain ({len(documents)} total)\n\n"

    for i, doc in enumerate(documents, 1):
        md += f"""
### {i}. {doc['doc_id']}
- **Uploaded:** {doc['upload_time']}
- **Chunks:** {doc['chunks_created']}
- **File:** {doc['file_name']} ({doc['file_size']})
- **Uploader:** {doc['uploader_id'] or 'Anonymous'}

---
"""

    return md


def on_create_domain(
        domain_id: str,
        display_name: str,
        description: str,
        collection_name: str
) -> tuple[str, str, List[str]]:
    """Create a new domain configuration."""
    try:
        # Validate inputs
        if not domain_id.strip():
            return "❌ Domain ID is required", "❌ Error", app_state.available_domains

        # Create config file path
        config_file = app_state.config_dir / "domains" / f"{domain_id}_domain.yaml"

        if config_file.exists():
            return f"❌ Domain '{domain_id}' already exists!", "❌ Error", app_state.available_domains

        # Create config from template
        config_data = {
            'name': domain_id,
            'display_name': display_name or domain_id.upper(),
            'description': description or f"Configuration for {domain_id} domain",
            'vector_store': {
                'provider': 'chromadb',
                'chromadb': {
                    'persist_directory': './data/chroma_db',
                    'collection_name': collection_name or f'{domain_id}_collection'
                }
            },
            'embeddings': {
                'provider': 'sentence_transformers',
                'model_name': 'all-MiniLM-L6-v2',
                'device': 'cpu',
                'batch_size': 32,
                'normalize_embeddings': True
            },
            'chunking': {
                'strategy': 'recursive',
                'recursive': {
                    'chunk_size': 500,
                    'overlap': 50
                },
                'semantic': {
                    'similarity_threshold': 0.7,
                    'max_chunk_size': 1000
                }
            },
            'retrieval': {
                'strategy': 'hybrid',
                'alpha': 0.7,
                'top_k': 10,
                'enable_metadata_filtering': True,
                'normalize_scores': True
            },
            'security': {
                'allowed_file_types': ['pdf', 'docx', 'txt'],
                'max_file_size_mb': 50
            }
        }

        # Write config file
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        # Refresh domains
        app_state.refresh_domains()

        logger.info(f"✅ Created domain: {domain_id}")

        success_md = f"""
## ✅ Domain Created Successfully!

**Domain ID:** `{domain_id}`  
**Display Name:** {display_name}  
**Config File:** `{config_file.name}`

The new domain is now available in the dropdown.
        """

        return success_md, f"✅ Created domain: **{domain_id}**", app_state.available_domains

    except Exception as e:
        error_msg = f"❌ Failed to create domain: {str(e)}"
        logger.error(error_msg)
        return error_msg, "❌ Error", app_state.available_domains


def on_delete_domain(domain_to_delete: str, confirm: bool) -> tuple[str, str, List[str]]:
    """Delete a domain configuration."""
    try:
        if not confirm:
            return "⚠️ Please check the confirmation box to delete the domain.", "❌ Not confirmed", app_state.available_domains

        if not domain_to_delete:
            return "❌ Please select a domain to delete.", "❌ Error", app_state.available_domains

        # Delete config file
        config_file = app_state.config_dir / "domains" / f"{domain_to_delete}_domain.yaml"

        if not config_file.exists():
            return f"❌ Domain config not found: {config_file}", "❌ Error", app_state.available_domains

        # Delete the file
        config_file.unlink()

        # Clear pipeline cache
        if domain_to_delete in app_state.pipelines:
            del app_state.pipelines[domain_to_delete]

        # Clear document tracking
        if domain_to_delete in app_state.uploaded_documents:
            del app_state.uploaded_documents[domain_to_delete]

        # Refresh domains
        app_state.refresh_domains()

        logger.info(f"✅ Deleted domain: {domain_to_delete}")

        success_md = f"""
## ✅ Domain Deleted Successfully!

**Deleted Domain:** `{domain_to_delete}`  
**Config File Removed:** `{config_file.name}`

⚠️ **Note:** The vector store data still exists in `./data/chroma_db/`.
To completely remove the data, manually delete the collection directory.
        """

        return success_md, f"✅ Deleted domain: **{domain_to_delete}**", app_state.available_domains

    except Exception as e:
        error_msg = f"❌ Failed to delete domain: {str(e)}"
        logger.error(error_msg)
        return error_msg, "❌ Error", app_state.available_domains


# =============================================================================
# UI Functions - Document Management
# =============================================================================

def on_document_upload(
        file,
        doc_id: str,
        uploader_id: str
) -> tuple[str, str, str, str]:
    """Enhanced document upload - NON-GENERATOR VERSION (more reliable)."""
    if not app_state.current_domain:
        return (
            "⚠️ **Please select a domain first in Tab 1!**",
            "❌ No domain selected",
            "",
            ""
        )

    if file is None:
        return (
            "⚠️ **Please upload a file!**",
            "❌ No file uploaded",
            "",
            ""
        )

    if not doc_id.strip():
        return (
            "⚠️ **Please provide a Document ID!**",
            "❌ Document ID required",
            "",
            ""
        )

    try:
        pipeline = app_state.get_pipeline(app_state.current_domain)
        file_path = file.name
        file_size = Path(file_path).stat().st_size

        # Build processing log
        log_parts = []

        log_parts.append(f"""
### 📄 Processing Document

**File:** `{Path(file_path).name}`  
**Size:** {format_file_size(file_size)}  
**Document ID:** `{doc_id}`  
**Domain:** `{app_state.current_domain}`  
**Uploader:** `{uploader_id or 'Anonymous'}`

---
""")

        # Step 1: Extract text
        logger.info(f"Extracting text from {file_path}")
        log_parts.append("⏳ **Step 1:** Extracting text from file...\n")

        text, file_type = extract_text_from_file(file_path)

        log_parts.append(f"✅ **Step 1 Complete:** Extracted {len(text)} characters from {file_type}\n\n")

        # Step 2: Process through pipeline
        log_parts.append(f"⏳ **Step 2:** Processing with {pipeline.config.chunking.strategy} chunking...\n")

        logger.info(f"Processing document through pipeline: {doc_id}")
        result = pipeline.process_document(
            file_path=file_path,
            doc_id=doc_id,
            uploader_id=uploader_id or None
        )

        if result['success']:
            # Add to tracking
            doc_info = {
                'doc_id': doc_id,
                'file_name': Path(file_path).name,
                'file_size': format_file_size(file_size),
                'chunks_created': result['chunks_created'],
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'uploader_id': uploader_id or 'Anonymous'
            }
            app_state.add_document_record(app_state.current_domain, doc_info)

            log_parts.append(f"""
✅ **Step 2 Complete:** Created {result['chunks_created']} chunks

⏳ **Step 3:** Generating embeddings with `{result['embedding_model']}`...
✅ **Step 3 Complete:** Embeddings generated

⏳ **Step 4:** Storing in `{result['vector_store']}` vector store...
✅ **Step 4 Complete:** All chunks stored successfully

---

## ✅ Upload Successful!

**Processing Time:** {result['processing_time']:.2f}s  
**Chunks Created:** {result['chunks_created']}  
**Status:** 🎉 Document is now searchable in the vector store!

You can now query this document in **Tab 3: Query Documents**
""")

            # Build metrics
            metrics_md = f"""
### 📊 Processing Metrics

| Metric | Value |
|--------|-------|
| **Chunks Created** | {result['chunks_created']} |
| **Processing Time** | {result['processing_time']:.2f}s |
| **Characters Extracted** | {len(text):,} |
| **Chunking Strategy** | {result['chunking_strategy']} |
| **Embedding Model** | {result['embedding_model']} |
| **Vector Store** | {result['vector_store']} |
| **File Hash** | `{result['file_hash'][:16]}...` |
"""

            status_msg = f"✅ **SUCCESS:** '{doc_id}' uploaded successfully ({result['chunks_created']} chunks created)"

            # Update document list
            docs_md = format_document_list(app_state.current_domain)

            # Combine all log parts
            full_log = "".join(log_parts)

            logger.info(f"✅ Document uploaded successfully: {doc_id}")

            return (full_log, status_msg, metrics_md, docs_md)

        else:
            # Processing failed
            log_parts.append(f"""
❌ **Processing Failed**

**Error Type:** {result.get('error_type', 'Unknown')}  
**Error Message:** {result.get('error', 'Unknown error occurred')}

**Troubleshooting:**
- Check if the file format is supported (PDF, DOCX, TXT)
- Verify the file is not corrupted
- Check the terminal/console for detailed error logs
""")

            full_log = "".join(log_parts)
            error_msg = f"❌ Processing failed: {result.get('error', 'Unknown error')}"

            logger.error(f"Document processing failed: {result.get('error')}")

            return (full_log, error_msg, "", "")

    except Exception as e:
        error_msg = f"❌ Unexpected Error: {str(e)}"
        logger.error(f"Upload error: {e}", exc_info=True)

        error_log = f"""
## ❌ Upload Error

**Error:** {str(e)}

**What to check:**
1. Is a domain selected in Tab 1?
2. Is the file format supported?
3. Check the terminal for detailed error logs

**File Info:**
- Path: {file.name if file else 'None'}
- Document ID: {doc_id}
- Domain: {app_state.current_domain}
"""

        return (error_log, error_msg, "", "")


def on_delete_document(doc_id_to_delete: str, confirm: bool) -> tuple[str, str]:
    """Delete a document from vector store."""
    try:
        if not confirm:
            return "⚠️ Check the confirmation box to delete.", "❌ Not confirmed"

        if not app_state.current_domain:
            return "❌ Select a domain first.", "❌ No domain"

        if not doc_id_to_delete.strip():
            return "❌ Enter a document ID to delete.", "❌ No doc ID"

        # Delete from vector store
        pipeline = app_state.get_pipeline(app_state.current_domain)
        result = pipeline.delete_document(doc_id_to_delete)

        if result['success']:
            # Remove from tracking
            app_state.remove_document_record(app_state.current_domain, doc_id_to_delete)

            success_md = f"""
## ✅ Document Deleted

**Document ID:** `{doc_id_to_delete}`  
**Domain:** `{app_state.current_domain}`

All chunks for this document have been removed from the vector store.
"""

            return success_md, f"✅ Deleted: **{doc_id_to_delete}**"
        else:
            return f"❌ Error: {result.get('error')}", "❌ Delete failed"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg


# =============================================================================
# UI Functions - Query
# =============================================================================

def on_query_submit(query: str, top_k: int, domain_filter: str) -> tuple[str, str]:
    """Handle query with results."""
    if not query.strip():
        return "⚠️ **Enter a query!**", "❌ Empty query"

    query_domain = domain_filter if domain_filter != "All Domains" else app_state.current_domain

    if not query_domain:
        return "⚠️ **Select a domain first!**", "❌ No domain"

    try:
        pipeline = app_state.get_pipeline(query_domain)

        # Generate embedding and search
        query_embedding = pipeline.embedder.embed_texts([query])[0]

        start_time = time.time()
        results = pipeline.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters={"domain": query_domain} if query_domain else None
        )
        search_time = time.time() - start_time

        # Format results
        if not results:
            results_md = f"""
## 🔍 Query Results

**Query:** "{query}"  
**Domain:** {query_domain}

---

⚠️ **No results found.** Try uploading documents first.
"""
            status_msg = "⚠️ No results"
        else:
            results_md = f"""
## 🔍 Query Results

**Query:** "{query}"  
**Domain:** {query_domain}  
**Found:** {len(results)} results ({search_time:.3f}s)

---

"""
            for i, result in enumerate(results, 1):
                score = result.get('score', result.get('distance', 0))
                metadata = result['metadata']

                results_md += f"""
### 📄 Result {i} • Score: {score:.4f}

**Document:** `{metadata.get('doc_id', 'Unknown')}`  
**Domain:** `{metadata.get('domain', 'Unknown')}`

**Content:**
> {result['document'][:500]}...

---
"""

            status_msg = f"✅ Found {len(results)} results in {search_time:.3f}s"

        return results_md, status_msg

    except Exception as e:
        error_msg = f"❌ Search Error: {str(e)}"
        return f"## {error_msg}", error_msg


# =============================================================================
# Gradio UI Layout
# =============================================================================

def create_ui():
    """Create enhanced Gradio UI."""

    with gr.Blocks(title="Multi-Domain RAG System", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # Multi-Domain Document Intelligence Platform

        **Complete document management with domain administration**
        """)

        # Shared status bar
        status_bar = gr.Markdown("ℹ️ Select a domain to get started")

        # =========== TAB 1: Domain Selection & Info ===========

        with gr.Tab("1️⃣ Domain Selection"):
            gr.Markdown("## 🏢 Select Your Domain")

            with gr.Row():
                with gr.Column(scale=1):
                    domain_dropdown = gr.Dropdown(
                        choices=app_state.available_domains,
                        label="Available Domains",
                        value=app_state.available_domains[0] if app_state.available_domains else None
                    )
                    domain_select_btn = gr.Button("🔄 Load Domain", variant="primary")

                with gr.Column(scale=2):
                    domain_info = gr.Markdown("Select a domain...")

            with gr.Accordion("📚 Documents in this Domain", open=True):
                domain_docs_list = gr.Markdown("Select a domain to see documents")

        # =========== TAB 2: Document Upload ===========

        with gr.Tab("2️⃣ Document Upload"):
            gr.Markdown("## 📤 Upload & Process Documents")

            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="Upload Document",
                        file_types=[".pdf", ".docx", ".txt"]
                    )
                    doc_id_input = gr.Textbox(label="Document ID", placeholder="doc_123")
                    uploader_id_input = gr.Textbox(label="Uploader ID (Optional)", placeholder="user@example.com")
                    process_btn = gr.Button("🚀 Process Document", variant="primary", size="lg")

                with gr.Column(scale=2):
                    processing_log = gr.Markdown("Upload a document to see processing...")
                    with gr.Accordion("📊 Metrics", open=False):
                        metrics_display = gr.Markdown("")

            with gr.Accordion("📚 Updated Document List", open=True):
                upload_docs_list = gr.Markdown("")

        # =========== TAB 3: Query ===========

        with gr.Tab("3️⃣ Query Documents"):
            gr.Markdown("## 🔍 Search Your Documents")

            with gr.Row():
                with gr.Column(scale=1):
                    query_input = gr.Textbox(label="Enter Query", lines=3, placeholder="What is RAG?")
                    with gr.Row():
                        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top K Results")
                        domain_filter_dropdown = gr.Dropdown(
                            choices=["All Domains"] + app_state.available_domains,
                            value="All Domains",
                            label="Filter by Domain"
                        )
                    search_btn = gr.Button("🔍 Search", variant="primary", size="lg")

                with gr.Column(scale=2):
                    search_results = gr.Markdown("Enter a query to search...")

        # =========== TAB 4: Create Domain ===========

        with gr.Tab("4️⃣ Create Domain"):
            gr.Markdown("## ➕ Create New Domain")

            with gr.Row():
                with gr.Column():
                    new_domain_id = gr.Textbox(label="Domain ID", placeholder="engineering")
                    new_display_name = gr.Textbox(label="Display Name", placeholder="Engineering")
                    new_description = gr.Textbox(label="Description", placeholder="Engineering documentation and APIs",
                                                 lines=3)
                    new_collection_name = gr.Textbox(label="Collection Name", placeholder="engineering_collection")
                    create_domain_btn = gr.Button("➕ Create Domain", variant="primary")

                with gr.Column():
                    create_domain_result = gr.Markdown("Fill the form and click Create")

        # =========== TAB 5: Delete Domain ===========

        with gr.Tab("5️⃣ Delete Domain"):
            gr.Markdown("## 🗑️ Delete Domain")

            with gr.Row():
                with gr.Column():
                    delete_domain_dropdown = gr.Dropdown(
                        choices=app_state.available_domains,
                        label="Select Domain to Delete"
                    )
                    delete_confirm_checkbox = gr.Checkbox(label="⚠️ I confirm deletion", value=False)
                    delete_domain_btn = gr.Button("🗑️ Delete Domain", variant="stop")

                with gr.Column():
                    delete_domain_result = gr.Markdown("⚠️ **Warning:** This cannot be undone!")

        # =========== TAB 6: Manage Documents ===========

        with gr.Tab("6️⃣ Manage Documents"):
            gr.Markdown("## 📁 Document Management")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🗑️ Delete Document")
                    delete_doc_id = gr.Textbox(label="Document ID to Delete", placeholder="doc_123")
                    delete_doc_confirm = gr.Checkbox(label="⚠️ Confirm deletion", value=False)
                    delete_doc_btn = gr.Button("🗑️ Delete Document", variant="stop")
                    delete_doc_result = gr.Markdown("")

                with gr.Column(scale=2):
                    gr.Markdown("### 📚 All Documents")
                    manage_docs_list = gr.Markdown("Select a domain to see documents")



                # =========== Event Handlers ===========

        # Domain selection
        domain_select_btn.click(
            fn=on_domain_select,
            inputs=[domain_dropdown],
            outputs=[domain_info, status_bar, domain_docs_list]
        )

        domain_dropdown.change(
            fn=on_domain_select,
            inputs=[domain_dropdown],
            outputs=[domain_info, status_bar, domain_docs_list]
        )

        # Document upload
        process_btn.click(
            fn=on_document_upload,
            inputs=[file_upload, doc_id_input, uploader_id_input],
            outputs=[processing_log, status_bar, metrics_display, upload_docs_list]
        )

        # Query
        search_btn.click(
            fn=on_query_submit,
            inputs=[query_input, top_k_slider, domain_filter_dropdown],
            outputs=[search_results, status_bar]
        )

        # Create domain
        create_domain_btn.click(
            fn=on_create_domain,
            inputs=[new_domain_id, new_display_name, new_description, new_collection_name],
            outputs=[create_domain_result, status_bar, domain_dropdown]
        )

        # Delete domain
        delete_domain_btn.click(
            fn=on_delete_domain,
            inputs=[delete_domain_dropdown, delete_confirm_checkbox],
            outputs=[delete_domain_result, status_bar, domain_dropdown]
        )

        # Delete document
        delete_doc_btn.click(
            fn=on_delete_document,
            inputs=[delete_doc_id, delete_doc_confirm],
            outputs=[delete_doc_result, status_bar]
        )

        # Update document list in manage tab when domain changes
        domain_dropdown.change(
            fn=lambda d: format_document_list(d) if d else "",
            inputs=[domain_dropdown],
            outputs=[manage_docs_list]
        )

        gr.Markdown("""
        ---
        **💡 Tip:** Create domains → Upload documents → Query them!

        **🔧 Tech:** Python, LangChain, Sentence-Transformers, ChromaDB, Gradio
        """)

    return app


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starting Enhanced Multi-Domain RAG System")
    logger.info("=" * 70)

    if not app_state.available_domains:
        logger.error("❌ No domains found!")
        logger.info("Creating a default 'hr' domain...")

        # Create default domain
        try:
            on_create_domain("hr", "Human Resources", "HR policies and documents", "hr_collection")
            app_state.refresh_domains()
        except:
            logger.error("Failed to create default domain")
            exit(1)

    logger.info(f"✅ Found {len(app_state.available_domains)} domains:")
    for domain in app_state.available_domains:
        logger.info(f"   - {domain}")

    app = create_ui()

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )
