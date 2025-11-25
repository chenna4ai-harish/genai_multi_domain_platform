import gradio as gr
import uuid
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import yaml

from core.registry.component_registry import ComponentRegistry

# ====================================================================
# Playground config helpers
# ====================================================================

def load_config_list() -> List[str]:
    from core.playground_config_manager import PlaygroundConfigManager
    return [c["name"] for c in PlaygroundConfigManager.list_configs()]


def load_config(config_name: str) -> Dict[str, Any]:
    from core.playground_config_manager import PlaygroundConfigManager
    all_configs = PlaygroundConfigManager.list_configs()
    match = next((c for c in all_configs if c["name"] == config_name), None)
    if not match:
        return {}
    return PlaygroundConfigManager.load_config(match["filename"])


def save_config(
    config_name,
    config_desc,
    vectorstore,
    distance_metric,
    collection_name,
    persist_dir,
    chunking_strategy,
    chunk_size,
    overlap,
    similarity_threshold,
    max_chunk_size,
    embedding_provider,
    embedding_model,
    device,
    batch_size,
    retrieval_strategies,
    top_k,
    hybrid_alpha,
    llm_provider,
    llm_model,
    temperature,
    max_tokens,
    session_id,  # not used for filename now
):
    from core.playground_config_manager import PlaygroundConfigManager

    today_tag = datetime.now().strftime("%d%m%Y")   # e.g. 25032025
    full_name = f"{config_name}_{today_tag}" if config_name else today_tag

    config = {
        "name": full_name,
        "description": config_desc,
        "vectorstore": {
            "provider": vectorstore,
            "distance_metric": distance_metric,
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
        },
        "retrieval": {
            "strategies": retrieval_strategies,
            "top_k": top_k,
            "hybrid": {"alpha": hybrid_alpha} if "Hybrid" in retrieval_strategies else {},
        },
        "llm_rerank": {
            "provider": llm_provider,
            "model_name": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }

    PlaygroundConfigManager.save_config(full_name, today_tag, config)
    return f"✅ Config **{full_name}** saved on `{today_tag}`."


def save_as_template(template_name: str, config_name: str, session_id: str):
    from core.playground_config_manager import PlaygroundConfigManager

    if not template_name:
        return "⚠️ Please enter a template name."

    all_configs = PlaygroundConfigManager.list_configs()
    match = next(
        (
            c for c in all_configs
            if c.get("name") == config_name
            or c.get("playground_name") == config_name
            or c.get("filename") == config_name
        ),
        None,
    )
    if not match:
        return f"⚠️ No config named **{config_name}** found to save as template."

    cfg = PlaygroundConfigManager.load_config(match["filename"])
    path = Path("configs/templates") / f"{template_name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f)

    return f"⭐ Template **{template_name}** created from config **{config_name}**."


# Simple demo pipeline runner (Phase-1 style)
def run_pipeline(
    user_query: str,
    config_name: str,
    session_id: str,
):
    if not user_query.strip():
        return "⚠️ Please enter a question to run.", [], "No query provided."

    answer = (
        f"Demo answer for: **{user_query}**\n\n"
        f"_Using config_ `{config_name or 'Current (unsaved) config'}` "
        f"_session_ `{session_id}`."
    )

    demo_chunks = [
        {"doc_id": 1, "score": 0.87, "snippet": "Chunk 1 text snippet..."},
        {"doc_id": 2, "score": 0.82, "snippet": "Chunk 2 text snippet..."},
    ]
    logs = (
        "Demo pipeline executed.\nSteps:\n"
        "1) Embed query\n2) Retrieve chunks\n3) Generate answer"
    )

    return answer, demo_chunks, logs


# ====================================================================
# DocumentService wiring (for upload + corpus + chunks)
# ====================================================================

service_cache: Dict[str, "DocumentService"] = {}


def get_service_for_config(config_name: str):
    """
    Map Playground config/domain to a DocumentService.

    TODO: adapt mapping to your real domain/config relationship.
    Currently assuming `DocumentService(domain_id=config_name)`.
    """
    if not config_name:
        raise ValueError("Please select a config/domain first")

    if config_name in service_cache:
        return service_cache[config_name]

    from core.services.document_service import DocumentService
    service = DocumentService(domain_id=config_name)
    service_cache[config_name] = service
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
                        placeholder="./vector_store",
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

            vectorstore_cfg = cfg.get("vectorstore", {}) or {}
            chunking_cfg = cfg.get("chunking", {}) or {}
            embeddings_cfg = cfg.get("embeddings", {}) or {}
            retrieval_cfg = cfg.get("retrieval", {}) or {}

            vs_provider_val = vectorstore_cfg.get("provider", default_vectorstore)
            vs_distance_val = vectorstore_cfg.get("distance_metric", default_distance_metric)
            vs_collection_val = vectorstore_cfg.get("collection_name", "")
            vs_persist_val = vectorstore_cfg.get("persist_directory", "")

            strategy_val = chunking_cfg.get("strategy", default_chunking_strategy)
            strategy_params = chunking_cfg.get(strategy_val, {}) or {}
            chunk_size_val = strategy_params.get("chunk_size", 500)
            overlap_val = strategy_params.get("overlap", 50)
            similarity_threshold_val = strategy_params.get("similarity_threshold", 0.7)
            max_chunk_size_val = strategy_params.get("max_chunk_size", 1000)

            emb_provider_val = embeddings_cfg.get("provider", default_provider)
            emb_model_val = embeddings_cfg.get("model_name", default_model)
            device_val = embeddings_cfg.get("device", default_device)
            batch_size_val = embeddings_cfg.get("batch_size", 32)

            retrieval_strats_val = retrieval_cfg.get(
                "strategies", default_retrieval_strategies
            )
            top_k_val = retrieval_cfg.get("top_k", 10)
            hybrid_cfg = retrieval_cfg.get("hybrid", {}) or {}
            hybrid_alpha_val = hybrid_cfg.get("alpha", 0.5)

            return (
                status,
                cfg.get("name", selected_name),  # config_name
                cfg.get("description", ""),      # config_desc
                vs_provider_val,
                vs_distance_val,
                vs_collection_val,
                vs_persist_val,
                strategy_val,
                chunk_size_val,
                overlap_val,
                similarity_threshold_val,
                max_chunk_size_val,
                emb_provider_val,
                emb_model_val,
                device_val,
                batch_size_val,
                retrieval_strats_val,
                top_k_val,
                hybrid_alpha_val,
            )

        load_btn.click(
            on_load_config,
            inputs=[config_selector, session_id],
            outputs=[
                config_status,
                config_name,
                config_desc,
                vectorstore,
                distance_metric,
                collection_name,
                persist_dir,
                chunking_strategy,
                chunk_size,
                overlap,
                similarity_threshold,
                max_chunk_size,
                embedding_provider,
                embedding_model,
                device,
                batch_size,
                retrieval_strategies,
                top_k,
                hybrid_alpha,
            ],
        )

        # -----------------------------------------------------------------
        # Save config
        # -----------------------------------------------------------------

        save_btn.click(
            save_config,
            inputs=[
                config_name,
                config_desc,
                vectorstore,
                distance_metric,
                collection_name,
                persist_dir,
                chunking_strategy,
                chunk_size,
                overlap,
                similarity_threshold,
                max_chunk_size,
                embedding_provider,
                embedding_model,
                device,
                batch_size,
                retrieval_strategies,
                top_k,
                hybrid_alpha,
                llm_provider,
                llm_model,
                temperature,
                max_tokens,
                session_id,
            ],
            outputs=config_status,
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
            doc_type: str,
            uploader_id: str,
            replace_existing: bool,
        ):
            if not selected_config:
                return "⚠️ Please select a config/domain in header dropdown.", []
            if file is None:
                return "⚠️ Please choose a file to upload.", []

            from core.services.document_service import ValidationError, ProcessingError

            try:
                service = get_service_for_config(selected_config)

                doc_id = f"{Path(file.name).stem}_{int(datetime.now().timestamp())}"
                metadata = {
                    "doc_id": doc_id,
                    "title": title or Path(file.name).stem,
                    "doc_type": doc_type or "playground",
                    "uploader_id": uploader_id or "playground_user",
                }

                result = service.upload_document(
                    file_obj=file,
                    metadata=metadata,
                    replace_existing=replace_existing,
                )

                metrics_table = [
                    ["Document ID", result.get("doc_id", doc_id)],
                    ["Chunks Ingested", result.get("chunks_ingested", 0)],
                    ["Embedding Model", result.get("embedding_model", "N/A")],
                    ["Chunking Strategy", result.get("chunking_strategy", "N/A")],
                    ["File Hash", result.get("file_hash", "N/A")],
                    ["Status", result.get("status", "success")],
                ]

                success_msg = (
                    f"✅ Uploaded `{file.name}` as `{result.get('doc_id', doc_id)}`\n\n"
                    f"- **Chunks:** {result.get('chunks_ingested', 0)}\n"
                    f"- **Embedding Model:** {result.get('embedding_model', 'N/A')}\n"
                    f"- **Chunking:** {result.get('chunking_strategy', 'N/A')}\n"
                )

                return success_msg, metrics_table

            except ValidationError as e:
                return f"❌ Validation error: {e}", []
            except ProcessingError as e:
                return f"❌ Processing error: {e}", []
            except Exception as e:
                return f"❌ Unexpected error: {e}", []

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
            outputs=[upload_status, upload_metrics],
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
