import gradio as gr
import uuid
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import yaml
import copy
import logging
import hashlib

logger = logging.getLogger(__name__)

# ONLY import PlaygroundConfigManager for playground UI
from core.playground_config_manager import PlaygroundConfigManager
from core.registry.component_registry import ComponentRegistry


# ============================================================================
# Backend Functions - ALL use PlaygroundConfigManager
# ============================================================================

def load_config_list() -> List[str]:
    """Return list of available playground configs."""
    return [c["name"] for c in PlaygroundConfigManager.list_configs()]


def load_config(config_name: str) -> Dict[str, Any]:
    """Load playground config by name."""
    all_configs = PlaygroundConfigManager.list_configs()
    match = next((c for c in all_configs if c["name"] == config_name), None)
    if not match:
        return {}
    return PlaygroundConfigManager.load_config(match["filename"])


def save_config(
        config_name, config_desc, vectorstore, distance_metric, collection_name,
        persist_dir, chunking_strategy, chunk_size, overlap, similarity_threshold,
        max_chunk_size, embedding_provider, embedding_model, device, batch_size,
        retrieval_strategies, top_k, hybrid_alpha, llm_provider, llm_model,
        temperature, max_tokens, session_id
):
    """Save playground config."""
    today_tag = datetime.now().strftime("%d%m%Y")
    full_name = f"{config_name}_{today_tag}" if config_name else today_tag

    config = {
        "name": full_name,
        "domain_id": full_name,  # ← ADD THIS
        "description": config_desc,
        "vector_store": {  # ← CHANGED from "vectorstore"
            "provider": vectorstore,
            "distance_metric": distance_metric,  # ← This might not be in schema, remove if needed
            "collection_name": collection_name,
            "persist_directory": persist_dir,
        },
        "chunking": {
            "strategy": chunking_strategy,
            chunking_strategy: {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "similarity_threshold": similarity_threshold,
                "max_chunk_size": max_chunk_size,
            },
        },
        "embeddings": {
            "provider": embedding_provider,
            "model_name": embedding_model,
            "device": device,
            "batch_size": batch_size,
            "normalize": True  # ← ADD THIS (required by schema)
        },
        "retrieval": {
            "strategies": retrieval_strategies,
            "top_k": top_k,
            "similarity": "cosine",  # ← ADD THIS (required by schema)
            "hybrid": {"alpha": hybrid_alpha} if "Hybrid" in retrieval_strategies else {},
        },
        "security": {  # ← ADD THIS (required by schema)
            "allowed_file_types": ["pdf", "docx", "txt"],
            "max_file_size_mb": 50
        },
        "llm_rerank": {
            "provider": llm_provider,
            "model_name": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }

    PlaygroundConfigManager.save_config(full_name, today_tag, config)
    return f"✅ Config **{full_name}** saved on {today_tag}."


def save_as_template(template_name: str, config_name: str, session_id: str):
    """Save playground config as template."""
    if not template_name:
        return "⚠️ Please enter a template name."

    all_configs = PlaygroundConfigManager.list_configs()
    match = next(
        (c for c in all_configs
         if c.get("name") == config_name
         or c.get("playground_name") == config_name
         or c.get("filename") == config_name),
        None
    )

    if not match:
        return f"❌ No config named '{config_name}' found to save as template."

    cfg = PlaygroundConfigManager.load_config(match["filename"])
    path = Path("configs/templates") / f"{template_name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(cfg, f)

    return f"⭐ Template **{template_name}** created from config **{config_name}**."


# ============================================================================
# Service Factory - Works with Playground Configs ONLY
# ============================================================================

service_cache: Dict[str, Any] = {}


def get_service_for_config(config_name: str):
    """
    Create DocumentService from playground config.
    Only works with configs/playground/, NOT configs/domains/.
    """
    if not config_name:
        raise ValueError("Please select a config first")

    if config_name in service_cache:
        return service_cache[config_name]

    from core.services.document_service import DocumentService

    # Find playground config
    pg_filename = PlaygroundConfigManager.find_config_by_name(config_name)
    if not pg_filename:
        raise FileNotFoundError(
            f"Playground config '{config_name}' not found. "
            f"Available: {[c['name'] for c in PlaygroundConfigManager.list_configs()]}"
        )

    # Load playground config
    pg_cfg = PlaygroundConfigManager.load_config(pg_filename)

    # Merge with global defaults
    pg_mgr = PlaygroundConfigManager()
    merged_cfg = pg_mgr.merge_with_global(pg_cfg)

    # Create synthetic domain ID for playground
    synth_domain_id = pg_cfg.get("playground_name") or config_name
    merged_cfg.setdefault("domain_id", synth_domain_id)
    merged_cfg.setdefault("name", synth_domain_id)

    # Validate as DomainConfig
    try:
        from core.config_manager import DomainConfig
        domain_config = DomainConfig(**merged_cfg)
    except Exception as e:
        logger.exception("Failed to validate playground config as DomainConfig")
        raise ValueError(f"Playground config '{config_name}' invalid: {e}")

    # Try to create service with DomainConfig
    try:
        service = DocumentService(domain_config=domain_config)
        service_cache[config_name] = service
        logger.info(f"DocumentService created from playground config '{config_name}'")
        return service
    except TypeError:
        # Fallback: write temp domain YAML
        logger.debug("DocumentService doesn't accept domain_config, writing temp domain file")
        temp_name = f"{synth_domain_id}_playground_temp"
        temp_domain_file = Path("configs/domains") / f"{temp_name}.yaml"

        if not temp_domain_file.exists():
            domain_dict = domain_config.model_dump() if hasattr(domain_config, 'model_dump') else domain_config.dict()
            with open(temp_domain_file, "w") as f:
                yaml.safe_dump(domain_dict, f)
            logger.info(f"Wrote temp domain file: {temp_domain_file}")

        service = DocumentService(domain_id=temp_name)
        service_cache[config_name] = service
        logger.info(f"DocumentService created from temp domain '{temp_name}'")
        return service
# ====================================================================
# UI / Playground Layout
# ====================================================================

def build_playground() -> gr.Blocks:
    # ---- Registry-driven options & defaults ----
    vectorstore_providers = ComponentRegistry.get_vectorstore_providers()
    distance_metrics = ComponentRegistry.get_distance_metrics()
    chunking_strategies = ComponentRegistry.get_chunking_strategies()
    embedding_providers = ComponentRegistry.get_embedding_providers()  # dict
    device_options = ComponentRegistry.get_device_options()
    retrieval_strategies_options = ComponentRegistry.get_retrieval_strategies()

    default_vectorstore = vectorstore_providers[0] if vectorstore_providers else None
    default_distance_metric = distance_metrics[0] if distance_metrics else None
    default_chunking_strategy = chunking_strategies[0] if chunking_strategies else "fixed"

    provider_choices = list(embedding_providers.keys())
    default_provider = provider_choices[0] if provider_choices else None
    default_models = embedding_providers.get(default_provider, []) if default_provider else []
    default_model = default_models[0] if default_models else None

    llm_providers = ComponentRegistry.get_llm_providers()
    llm_provider_choices = list(llm_providers.keys())
    default_llm_provider = llm_provider_choices[0] if llm_provider_choices else None
    default_llm_models = llm_providers.get(default_llm_provider, []) if default_llm_provider else []
    default_llm_model = default_llm_models[0] if default_llm_models else None

    default_device = device_options[0] if device_options else "cpu"
    default_retrieval_strategies = (
        [retrieval_strategies_options[0]] if retrieval_strategies_options else []
    )

    with gr.Blocks(title="RAG Playground - MVP") as demo:
        # ---- Session state ----
        raw_session_id = str(uuid.uuid4())[:8]
        session_id = gr.State(raw_session_id)

        # -----------------------------------------------------------------
        # Header
        # -----------------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🔧 RAG Playground")
                gr.Markdown(
                    "Configure, upload, and test your retrieval pipeline. "
                    "Designed for both devs and non-tech users."
                )
                gr.Markdown(f"**Session ID:** `{raw_session_id}`")
            with gr.Column(scale=1):
                with gr.Row():
                    config_selector = gr.Dropdown(
                        choices=load_config_list(),
                        label="Available configs",
                        value=None,
                        interactive=True,
                    )
                with gr.Row():
                    load_btn = gr.Button("📂 Load config")
                    save_btn = gr.Button("💾 Save config")
                    save_tpl_btn = gr.Button("⭐ Save as template")

        gr.Markdown("---")
        config_status = gr.Markdown("ℹ️ No config actions yet.")

        # -----------------------------------------------------------------
        # Main 50/50 split: LEFT = config, RIGHT = playground
        # -----------------------------------------------------------------
        with gr.Row():
            # =============== LEFT: Configuration (50%) ===============
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")

                config_name = gr.Textbox(
                    label="Config name",
                    placeholder="Example: Legal_Docs_Chroma_v1",
                )
                config_desc = gr.Textbox(
                    label="Description",
                    placeholder="Short description of what this config is for.",
                    lines=2,
                )

                # ---- Tab 1: Data & Vector Store ----
                with gr.Tab("1. Data & Vector Store"):
                    vectorstore = gr.Dropdown(
                        vectorstore_providers,
                        label="Vector store",
                        value=default_vectorstore,
                    )
                    distance_metric = gr.Dropdown(
                        distance_metrics,
                        label="Distance metric",
                        value=default_distance_metric,
                    )
                    collection_name = gr.Textbox(
                        label="Collection name",
                        placeholder="Name for this dataset in the vector store.",
                    )
                    persist_dir = gr.Textbox(
                        label="Persist directory",
                        placeholder="./vectorstore",
                    )

                # ---- Tab 2: Chunking ----
                with gr.Tab("2. Chunking"):
                    chunking_strategy = gr.Dropdown(
                        chunking_strategies,
                        label="Chunking strategy",
                        value=default_chunking_strategy,
                    )

                    chunk_size = gr.Slider(
                        100, 2000, value=500, step=50, label="Chunk Size", visible=True
                    )
                    overlap = gr.Slider(
                        0, 200, value=50, step=10, label="Overlap", visible=True
                    )
                    similarity_threshold = gr.Slider(
                        0.0,
                        1.0,
                        value=0.7,
                        step=0.01,
                        label="Similarity Threshold",
                        visible=False,
                    )
                    max_chunk_size = gr.Slider(
                        500,
                        3000,
                        value=1000,
                        step=50,
                        label="Max Chunk Size",
                        visible=False,
                    )

                # ---- Tab 3: Embeddings & Device ----
                with gr.Tab("3. Embeddings & Device"):
                    embedding_provider = gr.Dropdown(
                        provider_choices,
                        label="Embedding provider",
                        value=default_provider,
                    )
                    embedding_model = gr.Dropdown(
                        default_models,
                        label="Embedding model",
                        value=default_model,
                    )
                    device = gr.Dropdown(
                        device_options,
                        label="Device",
                        value=default_device,
                    )
                    with gr.Accordion("Performance tuning", open=False):
                        batch_size = gr.Slider(
                            1, 256, value=32, step=1, label="Batch size"
                        )

                # ---- Tab 4: Retrieval ----
                with gr.Tab("4. Retrieval"):
                    retrieval_strategies = gr.CheckboxGroup(
                        retrieval_strategies_options,
                        label="Retrieval strategies",
                        value=default_retrieval_strategies,
                    )
                    top_k = gr.Slider(
                        1, 50, value=10, step=1, label="Top K"
                    )
                    hybrid_alpha = gr.Slider(
                        0.0,
                        1.0,
                        value=0.5,
                        step=0.05,
                        label="Hybrid α (only used for Hybrid strategy)",
                    )
                    gr.Markdown(
                        "_Tip: Enable **Hybrid** above to make use of the α slider._"
                    )

                # ---- Tab 5: LLM ----
                with gr.Tab("5. LLM"):
                    llm_provider = gr.Dropdown(
                        llm_provider_choices,
                        label="LLM provider",
                        value=default_llm_provider,
                    )
                    llm_model = gr.Dropdown(
                        default_llm_models,
                        label="LLM model",
                        value=default_llm_model,
                    )
                    temperature = gr.Slider(
                        0.0, 1.0, value=0.2, step=0.01, label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        128, 4096, value=512, step=64, label="Max tokens"
                    )

            # =============== RIGHT: Playground (50%) ===============
            with gr.Column(scale=1):
                gr.Markdown("### 🧪 Playground")

                # ---- Tab A: Test query (demo) ----
                with gr.Tab("Test query"):
                    current_config_pill = gr.Markdown(
                        "Using config: _current session config_"
                    )
                    user_query = gr.Textbox(
                        label="Ask a question",
                        placeholder="Type your question here...",
                        lines=4,
                    )
                    run_btn = gr.Button("▶️ Run with current config")
                    answer_box = gr.Markdown(label="Answer")

                # ---- Tab B: Debug & Logs ----
                with gr.Tab("Debug & Logs"):
                    retrieved_chunks_df = gr.Dataframe(
                        headers=["doc_id", "score", "snippet"],
                        datatype=["number", "number", "str"],
                        row_count=2,
                        col_count=3,
                        interactive=False,
                        label="Retrieved chunks",
                    )
                    debug_log = gr.Textbox(
                        label="Logs",
                        lines=10,
                        interactive=False,
                    )

                # ---- Tab C: Corpus & Chunks (Upload + Explore) ----
                with gr.Tab("Corpus & Chunks"):
                    gr.Markdown("#### Corpus & Chunks")

                    with gr.Tab("Upload"):
                        gr.Markdown(
                            "Upload documents into the vector store using the selected config/domain."
                        )
                        upload_status = gr.Markdown("No upload yet.")
                        upload_metrics = gr.Dataframe(
                            headers=["Metric", "Value"],
                            value=[],
                            interactive=False,
                            row_count=0,
                        )

                        with gr.Row():
                            upload_file = gr.File(
                                label="Document file (.pdf, .docx, .txt)",
                                file_count="single",
                            )

                        with gr.Row():
                            upload_title = gr.Textbox(
                                label="Title",
                                placeholder="Optional, defaults to filename",
                            )
                            upload_doc_type = gr.Textbox(
                                label="Doc type",
                                placeholder="policy / faq / manual / etc.",
                            )

                        with gr.Row():
                            upload_uploader = gr.Textbox(
                                label="Uploader ID",
                                placeholder="e.g. harish@company.com",
                            )
                            upload_replace = gr.Checkbox(
                                label="Replace existing if doc_id already exists",
                                value=True,
                            )

                        upload_btn = gr.Button("⬆️ Upload & ingest")

                    with gr.Tab("Explore"):
                        gr.Markdown("Browse documents and their chunks for the selected config/domain.")
                        with gr.Row():
                            corpus_config_info = gr.Markdown(
                                "Using config/domain from header dropdown above."
                            )
                            refresh_docs_btn = gr.Button("🔄 Refresh documents", size="sm")

                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("##### Documents")
                                documents_status = gr.Markdown(
                                    value="No documents loaded yet."
                                )
                                documents_table = gr.Dataframe(
                                    headers=[
                                        "doc_id",
                                        "title",
                                        "doc_type",
                                        "uploader_id",
                                        "chunks",
                                        "last_seen",
                                        "deprecated",
                                    ],
                                    value=[],
                                    interactive=False,
                                    wrap=True,
                                    datatype=[
                                        "str",
                                        "str",
                                        "str",
                                        "str",
                                        "number",
                                        "str",
                                        "bool",
                                    ],
                                    row_count=0,
                                )

                            with gr.Column(scale=3):
                                gr.Markdown("##### Chunks")
                                selected_doc_id = gr.Textbox(
                                    label="Selected document ID",
                                    interactive=False,
                                )
                                chunks_table = gr.Dataframe(
                                    headers=[
                                        "chunk_id",
                                        "page",
                                        "char_start",
                                        "char_end",
                                        "snippet",
                                    ],
                                    value=[],
                                    interactive=False,
                                    wrap=True,
                                    row_count=0,
                                )
                                chunk_detail = gr.Textbox(
                                    label="Chunk detail",
                                    lines=10,
                                    interactive=False,
                                )
                                chunks_status = gr.Markdown(value="")

        # -----------------------------------------------------------------
        # Dynamic behaviors (models, chunking UI)
        # -----------------------------------------------------------------

        def update_embedding_models(provider: str):
            models = ComponentRegistry.get_embedding_providers().get(provider, [])
            return gr.update(
                choices=models,
                value=models[0] if models else None,
            )

        embedding_provider.change(
            update_embedding_models,
            inputs=[embedding_provider],
            outputs=[embedding_model],
        )

        def update_llm_models(provider: str):
            models = ComponentRegistry.get_llm_providers().get(provider, [])
            return gr.update(choices=models, value=models[0] if models else None)

        llm_provider.change(
            update_llm_models,
            inputs=[llm_provider],
            outputs=[llm_model],
        )

        def update_chunking_params(strategy: str):
            if strategy == "semantic":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

        chunking_strategy.change(
            update_chunking_params,
            inputs=[chunking_strategy],
            outputs=[chunk_size, overlap, similarity_threshold, max_chunk_size],
        )

        # -----------------------------------------------------------------
        # Load config
        # -----------------------------------------------------------------
        def on_load_config(selected_name: str, session_id_value: str):
            """Load playground config and populate UI fields."""
            if not selected_name:
                return (
                    "⚠️ Please select a config to load.",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            cfg = load_config(selected_name)
            status = f"📂 Loaded config **{selected_name}** for session `{session_id_value}`."

            # Extract config sections - try both field names for compatibility
            vectorstore_cfg = cfg.get("vector_store") or cfg.get("vectorstore") or {}
            chunking_cfg = cfg.get("chunking") or {}
            embeddings_cfg = cfg.get("embeddings") or {}
            retrieval_cfg = cfg.get("retrieval") or {}

            # === VECTOR STORE (FIX: Extract nested config) ===
            vs_provider_val = vectorstore_cfg.get("provider", default_vectorstore)

            # CRITICAL FIX: Get provider-specific nested config
            # Example: vectorstore.chromadb.collection_name
            provider_config = vectorstore_cfg.get(vs_provider_val, {})

            # Try nested first, then fallback to top-level (for compatibility)
            vs_collection_val = (
                    provider_config.get("collection_name") or
                    vectorstore_cfg.get("collection_name") or
                    ""
            )
            vs_persist_val = (
                    provider_config.get("persist_directory") or
                    vectorstore_cfg.get("persist_directory") or
                    ".vectorstore"
            )
            vs_distance_val = vectorstore_cfg.get("distance_metric", default_distance_metric)

            # === CHUNKING ===
            strategy_val = chunking_cfg.get("strategy", default_chunking_strategy)
            strategy_params = chunking_cfg.get(strategy_val, {}) or {}
            chunk_size_val = strategy_params.get("chunk_size", 500)
            overlap_val = strategy_params.get("overlap", 50)
            similarity_threshold_val = strategy_params.get("similarity_threshold", 0.7)
            max_chunk_size_val = strategy_params.get("max_chunk_size", 1000)

            # === EMBEDDINGS ===
            emb_provider_val = embeddings_cfg.get("provider", default_provider)
            emb_model_val = embeddings_cfg.get("model_name", default_model)
            device_val = embeddings_cfg.get("device", default_device)
            batch_size_val = embeddings_cfg.get("batch_size", 32)

            # === RETRIEVAL ===
            retrieval_strats_val = retrieval_cfg.get("strategies", default_retrieval_strategies)
            top_k_val = retrieval_cfg.get("top_k", 10)
            hybrid_cfg = retrieval_cfg.get("hybrid", {}) or {}
            hybrid_alpha_val = hybrid_cfg.get("alpha", 0.5)

            # Return all values in correct order matching the outputs
            return (
                status,  # config_status
                cfg.get("name", selected_name),  # config_name
                cfg.get("description", ""),  # config_desc
                vs_provider_val,  # vectorstore
                vs_distance_val,  # distance_metric
                vs_collection_val,  # collection_name ← NOW CORRECTLY EXTRACTED!
                vs_persist_val,  # persist_dir ← NOW CORRECTLY EXTRACTED!
                strategy_val,  # chunking_strategy
                chunk_size_val,  # chunk_size
                overlap_val,  # overlap
                similarity_threshold_val,  # similarity_threshold
                max_chunk_size_val,  # max_chunk_size
                emb_provider_val,  # embedding_provider
                emb_model_val,  # embedding_model
                device_val,  # device
                batch_size_val,  # batch_size
                retrieval_strats_val,  # retrieval_strategies
                top_k_val,  # top_k
                hybrid_alpha_val,  # hybrid_alpha
            )

        load_btn.click(
            on_load_config,
            inputs=[config_selector, session_id],
            outputs=[
                config_status,  # 1
                config_name,  # 2
                config_desc,  # 3
                vectorstore,  # 4
                distance_metric,  # 5
                collection_name,  # 6  ← CRITICAL!
                persist_dir,  # 7  ← CRITICAL!
                chunking_strategy,  # 8
                chunk_size,  # 9
                overlap,  # 10
                similarity_threshold,  # 11
                max_chunk_size,  # 12
                embedding_provider,  # 13
                embedding_model,  # 14
                device,  # 15
                batch_size,  # 16
                retrieval_strategies,  # 17
                top_k,  # 18
                hybrid_alpha,  # 19
            ],
        )

        # -----------------------------------------------------------------
        # Save as template
        # -----------------------------------------------------------------
        template_name_for_save = gr.Textbox(
            label="Template name (for 'Save as template')",
            placeholder="Example: Default_PDF_RAG",
            lines=1,
        )

        save_tpl_btn.click(
            save_as_template,
            inputs=[template_name_for_save, config_name, session_id],
            outputs=config_status,
        )

        # -----------------------------------------------------------------
        # Run pipeline (demo)
        # -----------------------------------------------------------------

        def on_run_pipeline(
            user_query_value: str,
            config_name_value: str,
            session_id_value: str,
        ):
            answer, chunks, logs = run_pipeline(
                user_query_value,
                config_name_value,
                session_id_value,
            )
            rows = [
                [c.get("doc_id"), c.get("score"), c.get("snippet")] for c in chunks
            ]
            pill_text = (
                f"Using config: _{config_name_value or 'Current (unsaved) config'}_"
            )
            return pill_text, answer, rows, logs

        run_btn.click(
            on_run_pipeline,
            inputs=[user_query, config_name, session_id],
            outputs=[current_config_pill, answer_box, retrieved_chunks_df, debug_log],
        )

        # -----------------------------------------------------------------
        # Upload document (Tab: Corpus & Chunks → Upload)
        # -----------------------------------------------------------------
        def on_upload_document(
                selected_config: str,
                file,
                title: str,
                doctype: str,
                uploader_id: str,
                replace_existing: bool,
        ):
            """Upload and ingest document into vector store."""

            # Validation
            if not selected_config:
                return "⚠️ Please select a config/domain in header dropdown.", []

            if file is None:
                return "⚠️ Please choose a file to upload.", []

            from core.services.document_service import ValidationError, ProcessingError

            try:
                # Get service for selected config
                service = get_service_for_config(selected_config)
                logger.info(f"Got service for config: {selected_config}")

                # Generate valid SHA-256 file hash
                file.seek(0)  # Reset file pointer
                file_content = file.read()
                file_hash = hashlib.sha256(file_content).hexdigest()  # ← CRITICAL FIX!
                file.seek(0)  # Reset for upload

                logger.info(f"File hash generated: {file_hash[:16]}...")

                # Generate doc_id
                doc_id = f"{Path(file.name).stem}_{int(datetime.now().timestamp())}"

                # Prepare metadata with ALL required fields
                metadata = {
                    "doc_id": doc_id,
                    "title": title or Path(file.name).stem,
                    "doc_type": doctype or "playground",
                    "uploader_id": uploader_id or "playground_user",
                    "domain": selected_config,  # ← ADD THIS
                    "source_file_path": file.name,  # ← ADD THIS
                    "file_hash": file_hash,  # ← ADD THIS (valid SHA-256)
                }

                logger.info(f"Uploading document with metadata: {metadata}")

                # Upload document
                result = service.upload_document(
                    file_obj=file,
                    metadata=metadata,
                    replace_existing=replace_existing,
                )

                logger.info(f"Upload result: {result}")

                # Prepare metrics table
                metrics_table = [
                    ["Document ID", result.get("doc_id", doc_id)],
                    ["Chunks Ingested", str(result.get("chunks_ingested", 0))],
                    ["Embedding Model", result.get("embedding_model", "N/A")],
                    ["Chunking Strategy", result.get("chunking_strategy", "N/A")],
                    ["File Hash", file_hash[:16] + "..."],  # Show first 16 chars
                    ["Status", result.get("status", "success")],
                ]

                # Prepare success message
                success_msg = (
                    f"✅ **Upload Successful!**\n\n"
                    f"📄 **File:** {file.name}\n"
                    f"🆔 **Doc ID:** {result.get('doc_id', doc_id)}\n"
                    f"📦 **Chunks:** {result.get('chunks_ingested', 0)}\n"
                    f"🤖 **Model:** {result.get('embedding_model', 'N/A')}\n"
                    f"✂️ **Chunking:** {result.get('chunking_strategy', 'N/A')}"
                )

                return success_msg, metrics_table

            except ValidationError as e:
                error_msg = f"❌ **Validation Error:**\n\n{str(e)}"
                logger.error(f"Validation error: {e}")
                return error_msg, []

            except ProcessingError as e:
                error_msg = f"❌ **Processing Error:**\n\n{str(e)}"
                logger.error(f"Processing error: {e}")
                return error_msg, []

            except Exception as e:
                error_msg = f"❌ **Unexpected Error:**\n\n{str(e)}"
                logger.exception(f"Upload failed for {selected_config}")
                return error_msg, []

        upload_btn.click(
            fn=on_upload_document,
            inputs=[
                config_selector,
                upload_file,
                upload_title,
                upload_doc_type,
                upload_uploader,
                upload_replace,
            ],
            outputs=[
                upload_status,  # Status message
                upload_metrics,  # Metrics table
            ],
        )
        # -----------------------------------------------------------------
        # Corpus Explorer (Tab: Corpus & Chunks → Explore)
        # -----------------------------------------------------------------

        def on_refresh_documents(selected_config: str):
            if not selected_config:
                return ("⚠️ Please select a config/domain first.", [])

            try:
                service = get_service_for_config(selected_config)
                docs = service.list_documents(filters={"deprecated": False})

                rows: List[List[Any]] = []
                for d in docs:
                    rows.append(
                        [
                            d.get("doc_id"),
                            d.get("title"),
                            d.get("doc_type"),
                            d.get("uploader_id"),
                            d.get("chunk_count"),
                            d.get("last_seen"),
                            d.get("deprecated", False),
                        ]
                    )

                if not rows:
                    status = f"ℹ️ No documents found for `{selected_config}`."
                else:
                    status = f"✅ Loaded {len(rows)} documents for `{selected_config}`."

                return status, rows

            except Exception as e:
                return (f"❌ Failed to load documents: `{e}`", [])

        refresh_docs_btn.click(
            fn=on_refresh_documents,
            inputs=[config_selector],
            outputs=[documents_status, documents_table],
        )

        def on_select_document(evt, docs_table_data, selected_config: str):
            if not docs_table_data:
                return "", [], "⚠️ No documents loaded.", ""

            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if idx is None or idx >= len(docs_table_data):
                return "", [], "⚠️ Invalid selection.", ""

            row = docs_table_data[idx]
            doc_id = row[0]

            if not selected_config:
                return doc_id, [], "⚠️ Please select a config/domain.", ""

            try:
                service = get_service_for_config(selected_config)
                chunks = service.list_chunks(doc_id=doc_id, limit=200)

                chunk_rows: List[List[Any]] = []
                for c in chunks:
                    md = c.get("metadata", {}) or {}
                    snippet = c.get("text", "")[:300].replace("\n", " ")
                    chunk_rows.append(
                        [
                            c.get("id"),
                            md.get("page_num"),
                            c.get("char_start"),
                            c.get("char_end"),
                            snippet,
                        ]
                    )

                if not chunk_rows:
                    status = f"ℹ️ No chunks found for document `{doc_id}`."
                else:
                    status = f"✅ Loaded {len(chunk_rows)} chunks for `{doc_id}`."

                return doc_id, chunk_rows, status, ""

            except Exception as e:
                return doc_id, [], f"❌ Failed to load chunks: `{e}`", ""

        documents_table.select(
            fn=on_select_document,
            inputs=[documents_table, config_selector],
            outputs=[selected_doc_id, chunks_table, chunks_status, chunk_detail],
        )

        def on_select_chunk(evt, chunks_table_data, selected_doc: str, selected_config: str):
            if not chunks_table_data:
                return "⚠️ No chunks loaded."

            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if idx is None or idx >= len(chunks_table_data):
                return "⚠️ Invalid chunk selection."

            chunk_id = chunks_table_data[idx][0]

            if not (selected_doc and selected_config):
                return "⚠️ Missing document/config selection."

            try:
                service = get_service_for_config(selected_config)
                chunks = service.list_chunks(doc_id=selected_doc, limit=None)

                for c in chunks:
                    if c.get("id") == chunk_id:
                        text = c.get("text", "")
                        md = c.get("metadata", {}) or {}
                        header = (
                            f"Chunk ID: {chunk_id}\n"
                            f"Page: {md.get('page_num')}\n"
                            f"Char range: {c.get('char_start')} - {c.get('char_end')}\n"
                            f"---\n\n"
                        )
                        return header + text

                return f"⚠️ Chunk `{chunk_id}` not found in latest list_chunks."

            except Exception as e:
                return f"❌ Failed to load chunk detail: `{e}`"

        chunks_table.select(
            fn=on_select_chunk,
            inputs=[chunks_table, selected_doc_id, config_selector],
            outputs=[chunk_detail],
        )

    return demo


if __name__ == "__main__":
    app = build_playground()
    app.launch()
