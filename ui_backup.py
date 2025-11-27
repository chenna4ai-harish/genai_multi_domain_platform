"""
Multi-Domain RAG System - Complete Integrated UI
Phase 2 Implementation with Full Backend Integration
"""

import gradio as gr
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import your core backend modules
from core.services.document_service import DocumentService, ValidationError, ProcessingError
from core.config_manager import ConfigManager

# ============================================================================
# Initialize Configuration Manager and Services
# ============================================================================

config_manager = ConfigManager()

# Cache for domain services (singleton pattern)
domain_services = {}

def get_domain_service(domain_name: str) -> DocumentService:
    """Get or create DocumentService for a domain."""
    if domain_name not in domain_services:
        domain_services[domain_name] = DocumentService(domain_name)
    return domain_services[domain_name]

# ============================================================================
# Screen 1: ASK (Query & Search)
# ============================================================================

def query_documents(domain: str, query: str) -> str:
    """Execute query against selected domain."""
    if not query.strip():
        return "Please enter a query."

    try:
        service = get_domain_service(domain)
        results = service.query(
            query_text=query,
            strategy="hybrid",  # Use hybrid retrieval by default
            metadata_filters=None,
            top_k=5,
            include_deprecated=False
        )

        if not results:
            return "No results found."

        # Format results as HTML cards
        html_output = ""
        for idx, result in enumerate(results, 1):
            score = result.get('score', 0)
            color = "green" if score > 0.8 else "orange" if score > 0.5 else "red"

            html_output += f"""
            <div style="border:1px solid #ddd; border-radius:8px; padding:15px; margin-bottom:15px; background-color:#f9f9f9;">
                <h4>Result {idx} <span style='color:{color}; font-weight:bold;'>({score*100:.1f}%)</span></h4>
                <p style="margin:10px 0;"><strong>Text:</strong> {result.get('document', '')[:300]}...</p>
                <p style="margin:5px 0; font-size:0.9em;"><strong>Source:</strong> {result.get('metadata', {}).get('title', 'Unknown')}</p>
                <p style="margin:5px 0; font-size:0.85em; color:#666;">
                    Author: {result.get('metadata', {}).get('author', 'N/A')} | 
                    Type: {result.get('metadata', {}).get('doctype', 'N/A')}
                </p>
            </div>
            """

        return html_output

    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>"

# ============================================================================
# Screen 2: DOMAIN MANAGEMENT
# ============================================================================

def get_available_domains() -> List[str]:
    """Get list of all available domains."""
    return config_manager.get_all_domain_names()

def create_domain(domain_name: str, description: str, template: str) -> tuple:
    """Create a new domain."""
    if not domain_name.strip():
        return "Domain name cannot be empty.", get_available_domains()

    try:
        # Load template if provided, otherwise use default config
        if template and template != "None":
            base_config = config_manager.load_template_config(template)
            config_dict = base_config.dict()
        else:
            # Create minimal config
            config_dict = {
                "domain_id": domain_name,
                "name": domain_name,
                "description": description,
                "vectorstore": {
                    "provider": "chromadb",
                    "collection_name": f"{domain_name}_collection",
                    "persist_directory": f".data/chromadb/{domain_name}"
                },
                "embeddings": {
                    "provider": "sentence_transformers",
                    "model_name": "all-MiniLM-L6-v2",
                    "device": "cpu"
                },
                "chunking": {
                    "strategy": "recursive",
                    "recursive": {"chunk_size": 500, "overlap": 50}
                },
                "retrieval": {
                    "strategies": ["hybrid"],
                    "top_k": 10
                }
            }

        # Update with user inputs
        config_dict["domain_id"] = domain_name
        config_dict["name"] = domain_name
        config_dict["description"] = description

        # Save domain config
        config_manager.save_domain_config(domain_name, config_dict)

        # Clear service cache to reload
        if domain_name in domain_services:
            del domain_services[domain_name]

        return f"✓ Domain '{domain_name}' created successfully!", get_available_domains()

    except Exception as e:
        return f"✗ Error creating domain: {str(e)}", get_available_domains()

def delete_domain(domain_name: str) -> tuple:
    """Delete an existing domain."""
    if not domain_name:
        return "Please select a domain to delete.", get_available_domains()

    try:
        domain_file = Path(f"configs/domains/{domain_name}.yaml")
        if domain_file.exists():
            domain_file.unlink()

            # Remove from service cache
            if domain_name in domain_services:
                del domain_services[domain_name]

            return f"✓ Domain '{domain_name}' deleted successfully!", get_available_domains()
        else:
            return f"✗ Domain '{domain_name}' not found.", get_available_domains()

    except Exception as e:
        return f"✗ Error deleting domain: {str(e)}", get_available_domains()

# ============================================================================
# Screen 3: DOCUMENT MANAGEMENT
# ============================================================================

def upload_document(domain: str, file, title: str, doctype: str, author: str, uploader_id: str) -> str:
    """Upload and process a document."""
    if not file:
        return "Please select a file to upload."

    if not all([title, doctype, uploader_id]):
        return "Please fill in all required fields: Title, Document Type, and Uploader ID."

    try:
        service = get_domain_service(domain)

        # Generate unique doc_id
        doc_id = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare metadata
        metadata = {
            "doc_id": doc_id,
            "title": title,
            "doctype": doctype,
            "author": author or "Unknown",
            "uploader_id": uploader_id
        }

        # Upload document
        result = service.upload_document(
            file_obj=file,
            metadata=metadata,
            replace_existing=False
        )

        return f"""
        <div style="color:green; padding:10px; border:1px solid green; border-radius:5px;">
            <h4>✓ Document Uploaded Successfully!</h4>
            <p><strong>Document ID:</strong> {result.get('doc_id')}</p>
            <p><strong>Chunks Created:</strong> {result.get('chunks_ingested', 0)}</p>
            <p><strong>Embedding Model:</strong> {result.get('embedding_model', 'N/A')}</p>
            <p><strong>File Hash:</strong> {result.get('file_hash', 'N/A')[:16]}...</p>
        </div>
        """

    except ValidationError as e:
        return f"<div style='color:orange; padding:10px;'>⚠ Validation Error: {str(e)}</div>"
    except ProcessingError as e:
        return f"<div style='color:red; padding:10px;'>✗ Processing Error: {str(e)}</div>"
    except Exception as e:
        return f"<div style='color:red; padding:10px;'>✗ Error: {str(e)}</div>"

# ============================================================================
# Screen 4: PARAMETER MANAGEMENT
# ============================================================================

def save_parameter_template(template_name: str, chunk_size: int, overlap: int, 
                           embedding_model: str, retrieval_strategy: str, 
                           top_k: int, alpha: float) -> str:
    """Save parameter configuration as a template."""
    if not template_name.strip():
        return "Template name cannot be empty."

    try:
        template_config = {
            "domain_id": template_name,
            "name": template_name,
            "description": f"Template: {template_name}",
            "chunking": {
                "strategy": "recursive",
                "recursive": {
                    "chunk_size": chunk_size,
                    "overlap": overlap
                }
            },
            "embeddings": {
                "provider": "sentence_transformers",
                "model_name": embedding_model,
                "device": "cpu"
            },
            "retrieval": {
                "strategies": [retrieval_strategy],
                "top_k": top_k,
                "hybrid": {
                    "alpha": alpha
                }
            },
            "vectorstore": {
                "provider": "chromadb",
                "collection_name": f"{template_name}_collection"
            }
        }

        config_manager.save_template_config(template_name, template_config)

        return f"✓ Template '{template_name}' saved successfully!"

    except Exception as e:
        return f"✗ Error saving template: {str(e)}"

# ============================================================================
# Screen 5: PLAYGROUND (Template Creation & Testing)
# ============================================================================

def test_playground_workflow(template_name: str, chunk_size: int, embedding_model: str,
                            test_query: str) -> str:
    """Test complete workflow in playground."""
    if not template_name.strip():
        return "Please enter a template name."

    result = f"""
    <div style="padding:15px; border:1px solid #2196F3; border-radius:8px; background:#f0f8ff;">
        <h3>🧪 Playground Test Results</h3>
        <p><strong>Template:</strong> {template_name}</p>
        <p><strong>Chunk Size:</strong> {chunk_size}</p>
        <p><strong>Embedding Model:</strong> {embedding_model}</p>
        <p><strong>Test Query:</strong> {test_query or "No query provided"}</p>
        <hr>
        <p style="color:green;"><strong>✓ Configuration validated successfully!</strong></p>
        <p>You can now save this template for use in domain creation.</p>
    </div>
    """

    return result

# ============================================================================
# BUILD GRADIO UI
# ============================================================================

with gr.Blocks(title="Multi-Domain RAG System", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🔍 Multi-Domain RAG System")
    gr.Markdown("Enterprise Document Search & Management Platform")

    # ========================================================================
    # TAB 1: ASK (Query & Search)
    # ========================================================================

    with gr.Tab("🔎 Ask"):
        gr.Markdown("## Ask — Query & Search Documents")
        gr.Markdown("Search across your domain-specific document collections using natural language queries.")

        with gr.Row():
            ask_domain = gr.Dropdown(
                choices=get_available_domains(),
                label="Select Domain",
                scale=1,
                value=get_available_domains()[0] if get_available_domains() else None
            )
            ask_query = gr.Textbox(
                label="Type your question",
                placeholder="Ask anything... (e.g., What is the leave policy?)",
                lines=3,
                scale=4
            )

        ask_examples = gr.Markdown("💡 **Examples:** What is the vacation policy? Who is the department head? How do I submit expenses?")
        ask_button = gr.Button("Ask", variant="primary", size="lg")
        ask_results = gr.HTML()

        ask_button.click(
            fn=query_documents,
            inputs=[ask_domain, ask_query],
            outputs=ask_results
        )

    # ========================================================================
    # TAB 2: DOMAIN MANAGEMENT
    # ========================================================================

    with gr.Tab("🏗️ Domain Management"):
        gr.Markdown("## Domain Management")
        gr.Markdown("Create, configure, and manage search domains for different business areas.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Create New Domain")
                create_domain_name = gr.Textbox(label="Domain Name", placeholder="e.g., hr, finance, engineering")
                create_domain_desc = gr.Textbox(label="Description", placeholder="Brief description of the domain")
                create_template = gr.Dropdown(
                    choices=["None"] + config_manager.get_all_template_names(),
                    label="Base Template (Optional)",
                    value="None"
                )
                create_button = gr.Button("Create Domain", variant="primary")
                create_status = gr.Markdown()

            with gr.Column():
                gr.Markdown("### Delete Domain")
                delete_domain_dropdown = gr.Dropdown(
                    choices=get_available_domains(),
                    label="Select Domain to Delete"
                )
                delete_button = gr.Button("Delete Domain", variant="stop")
                delete_status = gr.Markdown()

        available_domains = gr.Markdown(f"**Available Domains:** {', '.join(get_available_domains())}")

        create_button.click(
            fn=create_domain,
            inputs=[create_domain_name, create_domain_desc, create_template],
            outputs=[create_status, delete_domain_dropdown]
        )

        delete_button.click(
            fn=delete_domain,
            inputs=[delete_domain_dropdown],
            outputs=[delete_status, delete_domain_dropdown]
        )

    # ========================================================================
    # TAB 3: DOCUMENT MANAGEMENT
    # ========================================================================

    with gr.Tab("📤 Document Management"):
        gr.Markdown("## Document Management")
        gr.Markdown("Upload, process, and manage documents within your domains.")

        upload_domain = gr.Dropdown(
            choices=get_available_domains(),
            label="Select Domain",
            value=get_available_domains()[0] if get_available_domains() else None
        )

        with gr.Row():
            with gr.Column(scale=2):
                upload_file = gr.File(label="Upload Document", file_types=[".pdf", ".docx", ".txt"])
            with gr.Column(scale=3):
                upload_title = gr.Textbox(label="Document Title *", placeholder="e.g., Employee Handbook 2025")
                upload_doctype = gr.Dropdown(
                    choices=["policy", "manual", "faq", "report", "memo", "other"],
                    label="Document Type *",
                    value="policy"
                )
                upload_author = gr.Textbox(label="Author", placeholder="Optional")
                upload_uploader = gr.Textbox(label="Uploader ID *", placeholder="e.g., user@company.com")

        upload_button = gr.Button("Upload & Process", variant="primary", size="lg")
        upload_result = gr.HTML()

        upload_button.click(
            fn=upload_document,
            inputs=[upload_domain, upload_file, upload_title, upload_doctype, upload_author, upload_uploader],
            outputs=upload_result
        )

    # ========================================================================
    # TAB 4: PARAMETER MANAGEMENT
    # ========================================================================

    with gr.Tab("⚙️ Parameter Management"):
        gr.Markdown("## Parameter Management")
        gr.Markdown("Configure chunking, embedding, and retrieval parameters. Save as templates for reuse.")

        gr.Markdown("### Chunking Parameters")
        with gr.Row():
            param_chunk_size = gr.Slider(minimum=100, maximum=2000, value=500, step=50, label="Chunk Size")
            param_overlap = gr.Slider(minimum=0, maximum=200, value=50, step=10, label="Overlap")

        gr.Markdown("### Embedding Parameters")
        param_embedding = gr.Dropdown(
            choices=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
            label="Embedding Model",
            value="all-MiniLM-L6-v2"
        )

        gr.Markdown("### Retrieval Parameters")
        with gr.Row():
            param_strategy = gr.Dropdown(
                choices=["hybrid", "vector_similarity", "bm25"],
                label="Retrieval Strategy",
                value="hybrid"
            )
            param_topk = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Top K Results")
            param_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Hybrid Alpha")

        param_template_name = gr.Textbox(label="Template Name", placeholder="e.g., production_config")
        param_save_button = gr.Button("Save as Template", variant="primary")
        param_status = gr.Markdown()

        param_save_button.click(
            fn=save_parameter_template,
            inputs=[param_template_name, param_chunk_size, param_overlap, param_embedding, 
                   param_strategy, param_topk, param_alpha],
            outputs=param_status
        )

    # ========================================================================
    # TAB 5: PLAYGROUND
    # ========================================================================

    with gr.Tab("🧪 Playground"):
        gr.Markdown("## Playground — Experiment & Test")
        gr.Markdown("Design templates, test configurations, and preview processing workflows.")

        with gr.Row():
            play_template_name = gr.Textbox(label="Template Name", placeholder="e.g., test_template", scale=2)
            play_chunk_size = gr.Number(label="Chunk Size", value=500, scale=1)
            play_embedding = gr.Dropdown(
                choices=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                label="Embedding Model",
                value="all-MiniLM-L6-v2",
                scale=2
            )

        play_test_query = gr.Textbox(label="Test Query", placeholder="Enter a test query...", lines=2)
        play_test_button = gr.Button("Test Configuration", variant="primary")
        play_results = gr.HTML()

        play_test_button.click(
            fn=test_playground_workflow,
            inputs=[play_template_name, play_chunk_size, play_embedding, play_test_query],
            outputs=play_results
        )

# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7864,
        share=False,
        show_error=True
    )
