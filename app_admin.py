"""
app_admin.py — Admin Console (4-tab interface)

Tab 1 — Template Management   : view and list reusable config templates
Tab 2 — Domain Management     : create domains from templates (vector store initialized immediately)
Tab 3 — Document Management   : upload / browse / delete documents per domain
Tab 4 — Playground            : full RAG config tuner for experimentation

All operations go through the service layer (DocumentService, DomainService).
No business logic in this file.
"""

import gradio as gr
import yaml
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.config_manager import ConfigManager
from core.services.document_service import DocumentService
from core.services.domain_service import DomainService
from core.registry.component_registry import ComponentRegistry
from core.playground_config_manager import PlaygroundConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service helpers
# ---------------------------------------------------------------------------

_doc_service_cache: Dict[str, DocumentService] = {}


def _get_doc_service(domain_id: str) -> DocumentService:
    if domain_id not in _doc_service_cache:
        _doc_service_cache[domain_id] = DocumentService(domain_id)
    return _doc_service_cache[domain_id]


_domain_svc = DomainService()
_config_mgr = ConfigManager()


# ===========================================================================
# TAB 1 — TEMPLATE MANAGEMENT
# ===========================================================================

def list_templates_table() -> List[List]:
    templates = _domain_svc.list_templates()
    rows = []
    for t in templates:
        rows.append([
            t["template_name"],
            t["name"],
            t["description"],
            t["vectorstore_provider"],
            t["chunking_strategy"],
            t["embedding_provider"],
        ])
    return rows


def load_template_detail(template_name: str) -> str:
    if not template_name:
        return ""
    try:
        raw = _domain_svc.get_template_raw(template_name)
        return yaml.dump(raw, sort_keys=False, default_flow_style=False)
    except Exception as e:
        return f"Error: {e}"


def build_tab_templates() -> None:
    gr.Markdown("### Templates")
    gr.Markdown(
        "Templates are reusable base configurations stored in `configs/templates/`. "
        "When creating a domain, you pick a template and the system auto-populates the config."
    )

    refresh_tpl_btn = gr.Button("🔄 Refresh", size="sm")

    templates_table = gr.Dataframe(
        headers=["Template name", "Display name", "Description",
                 "Vector store", "Chunking", "Embeddings"],
        value=list_templates_table(),
        interactive=False,
        wrap=True,
        row_count=0,
    )

    gr.Markdown("#### Template details")
    selected_template_name = gr.Textbox(
        label="Template name (click a row above to select)",
        interactive=True,
        placeholder="e.g. test_template_hr_v1",
    )
    template_yaml_display = gr.Code(
        label="Template YAML",
        language="yaml",
        interactive=False,
    )

    refresh_tpl_btn.click(
        fn=lambda: list_templates_table(),
        outputs=[templates_table],
    )

    selected_template_name.change(
        fn=load_template_detail,
        inputs=[selected_template_name],
        outputs=[template_yaml_display],
    )


# ===========================================================================
# TAB 2 — DOMAIN MANAGEMENT
# ===========================================================================

def list_domains_table() -> List[List]:
    domains = _domain_svc.list_domains()
    rows = []
    for d in domains:
        vec_count = _domain_svc.get_domain_vector_count(d["domain_id"])
        rows.append([
            d["domain_id"],
            d["name"],
            d["description"],
            d["collection_name"],
            vec_count if vec_count is not None else "—",
            d["created_at"][:10] if d.get("created_at") else "",
        ])
    return rows


def load_template_for_form(template_name: str):
    """Auto-fill LLM provider/model dropdowns from selected template."""
    if not template_name:
        return gr.update(), gr.update()
    try:
        raw = _domain_svc.get_template_raw(template_name)
        llm_sec = raw.get("llm") or raw.get("llm_rerank") or {}
        provider = llm_sec.get("provider", "gemini")
        model = llm_sec.get("model_name", "gemini-1.5-flash")
        llm_providers = ComponentRegistry.get_llm_providers()
        models = llm_providers.get(provider, [])
        return (
            gr.update(value=provider),
            gr.update(choices=models, value=model if model in models else (models[0] if models else None)),
        )
    except Exception:
        return gr.update(), gr.update()


def create_domain_action(
    domain_id: str,
    domain_name: str,
    template_name: str,
    description: str,
    llm_provider: str,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
) -> tuple:
    if not domain_id or not template_name:
        return "⚠️ Domain ID and template are required.", list_domains_table()
    try:
        result = _domain_svc.create_domain(
            domain_id=domain_id,
            domain_name=domain_name or domain_id,
            template_name=template_name,
            description=description,
        )
        # Patch the llm section in the saved domain YAML
        domain_file = Path("configs/domains") / f"{result['domain_id']}.yaml"
        if domain_file.exists():
            with open(domain_file, "r") as f:
                domain_dict = yaml.safe_load(f) or {}
            domain_dict["llm"] = {
                "provider": llm_provider,
                "model_name": llm_model,
                "temperature": llm_temperature,
                "max_tokens": int(llm_max_tokens),
            }
            with open(domain_file, "w") as f:
                yaml.safe_dump(domain_dict, f, sort_keys=False, default_flow_style=False)

        msg = (
            f"✅ Domain **{result['domain_id']}** created.\n"
            f"- Collection: `{result['collection_name']}`\n"
            f"- Persist dir: `{result['persist_directory']}`\n"
            f"- Vectors: {result['vectors_in_collection']}\n"
            f"- LLM: `{llm_provider} / {llm_model}`"
        )
        return msg, list_domains_table()
    except Exception as e:
        return f"❌ {e}", list_domains_table()


def update_llm_models_for_provider(provider: str):
    models = ComponentRegistry.get_llm_providers().get(provider, [])
    return gr.update(choices=models, value=models[0] if models else None)


def build_tab_domains() -> None:
    llm_providers = ComponentRegistry.get_llm_providers()
    llm_provider_choices = list(llm_providers.keys())
    default_llm_provider = llm_provider_choices[0] if llm_provider_choices else "gemini"
    default_llm_models = llm_providers.get(default_llm_provider, [])
    default_llm_model = default_llm_models[0] if default_llm_models else None

    gr.Markdown("### Domain Management")

    # --- Create domain form ---
    gr.Markdown("#### Create New Domain")
    with gr.Row():
        with gr.Column(scale=1):
            domain_id_input = gr.Textbox(
                label="Domain ID",
                placeholder="e.g. hr, finance, legal (lowercase, no spaces)",
            )
            domain_name_input = gr.Textbox(
                label="Display Name",
                placeholder="e.g. Human Resources",
            )
            domain_desc_input = gr.Textbox(
                label="Description",
                placeholder="Short description of this domain",
                lines=2,
            )
            template_selector = gr.Dropdown(
                choices=_config_mgr.get_all_template_names(),
                label="Base Template",
                value=None,
            )

        with gr.Column(scale=1):
            gr.Markdown("**LLM Configuration** (used when users ask questions)")
            llm_provider_dd = gr.Dropdown(
                choices=llm_provider_choices,
                label="LLM Provider",
                value=default_llm_provider,
            )
            llm_model_dd = gr.Dropdown(
                choices=default_llm_models,
                label="LLM Model",
                value=default_llm_model,
            )
            llm_temperature_sl = gr.Slider(
                0.0, 1.0, value=0.2, step=0.05, label="Temperature"
            )
            llm_max_tokens_sl = gr.Slider(
                64, 4096, value=512, step=64, label="Max Tokens"
            )

    create_domain_btn = gr.Button("🚀 Create Domain + Initialize Vector Store", variant="primary")
    create_status = gr.Markdown("")

    gr.Markdown("---")

    # --- Existing domains list ---
    gr.Markdown("#### Existing Domains")
    refresh_domains_btn = gr.Button("🔄 Refresh", size="sm")
    domains_table = gr.Dataframe(
        headers=["Domain ID", "Name", "Description", "Collection", "Vectors", "Created"],
        value=list_domains_table(),
        interactive=False,
        wrap=True,
        row_count=0,
    )

    # Wire events
    template_selector.change(
        fn=load_template_for_form,
        inputs=[template_selector],
        outputs=[llm_provider_dd, llm_model_dd],
    )
    llm_provider_dd.change(
        fn=update_llm_models_for_provider,
        inputs=[llm_provider_dd],
        outputs=[llm_model_dd],
    )
    create_domain_btn.click(
        fn=create_domain_action,
        inputs=[
            domain_id_input, domain_name_input, template_selector, domain_desc_input,
            llm_provider_dd, llm_model_dd, llm_temperature_sl, llm_max_tokens_sl,
        ],
        outputs=[create_status, domains_table],
    )
    refresh_domains_btn.click(fn=list_domains_table, outputs=[domains_table])


# ===========================================================================
# TAB 3 — DOCUMENT MANAGEMENT
# ===========================================================================

def get_domain_list() -> List[str]:
    return _config_mgr.get_all_domain_names()


def on_upload_document(
    domain_id: str,
    file,
    title: str,
    doc_type: str,
    uploader_id: str,
    replace_existing: bool,
) -> tuple:
    if not domain_id:
        return "⚠️ Select a domain first.", []
    if file is None:
        return "⚠️ Select a file.", []
    try:
        svc = _get_doc_service(domain_id)
        doc_id = f"{Path(file.name).stem}_{int(datetime.now().timestamp())}"
        metadata = {
            "doc_id": doc_id,
            "title": title or Path(file.name).stem,
            "doc_type": doc_type or "document",
            "uploader_id": uploader_id or "admin",
        }
        result = svc.upload_document(
            file_obj=file,
            metadata=metadata,
            replace_existing=replace_existing,
        )
        rows = [
            ["Document ID", result.get("doc_id", doc_id)],
            ["Chunks ingested", result.get("chunks_ingested", 0)],
            ["Embedding model", result.get("embedding_model", "N/A")],
            ["Chunking strategy", result.get("chunking_strategy", "N/A")],
            ["File hash", result.get("file_hash", "N/A")[:16] + "..."],
            ["Status", result.get("status", "success")],
        ]
        return (
            f"✅ Uploaded `{Path(file.name).name}` — "
            f"{result.get('chunks_ingested', 0)} chunks ingested.",
            rows,
        )
    except Exception as e:
        return f"❌ {e}", []


def on_refresh_documents(domain_id: str) -> tuple:
    if not domain_id:
        return "⚠️ Select a domain.", []
    try:
        svc = _get_doc_service(domain_id)
        docs = svc.list_documents(filters={"deprecated": False})
        rows = [
            [
                d.get("doc_id"),
                d.get("title"),
                d.get("doc_type"),
                d.get("uploader_id"),
                d.get("chunk_count"),
                d.get("last_seen"),
            ]
            for d in docs
        ]
        status = f"✅ {len(rows)} document(s) in `{domain_id}`." if rows else f"ℹ️ No documents in `{domain_id}`."
        return status, rows
    except Exception as e:
        return f"❌ {e}", []


def on_select_document(evt: gr.SelectData, docs_data: List, domain_id: str) -> tuple:
    if not docs_data or evt.index is None:
        return "", [], ""
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx >= len(docs_data):
        return "", [], ""
    doc_id = docs_data[idx][0]
    if not domain_id or not doc_id:
        return doc_id, [], ""
    try:
        svc = _get_doc_service(domain_id)
        chunks = svc.list_chunks(doc_id=doc_id, limit=200)
        chunk_rows = []
        for c in chunks:
            md = c.get("metadata", {}) or {}
            snippet = c.get("text", "")[:200].replace("\n", " ")
            chunk_rows.append([c.get("id"), md.get("page_num"), snippet])
        status = f"✅ {len(chunk_rows)} chunks for `{doc_id}`."
        return doc_id, chunk_rows, status
    except Exception as e:
        return doc_id, [], f"❌ {e}"


def on_delete_document(domain_id: str, doc_id: str) -> tuple:
    if not domain_id or not doc_id.strip():
        return "⚠️ Select a domain and enter a document ID.", []
    try:
        svc = _get_doc_service(domain_id)
        svc.delete_document(doc_id.strip())
        return f"✅ Deleted `{doc_id}`.", on_refresh_documents(domain_id)[1]
    except Exception as e:
        return f"❌ {e}", []


def build_tab_documents() -> None:
    gr.Markdown("### Document Management")

    domain_dd = gr.Dropdown(
        choices=get_domain_list(),
        label="Domain",
        value=None,
        interactive=True,
    )
    refresh_domain_btn = gr.Button("🔄 Refresh domains", size="sm")

    with gr.Tab("Upload"):
        upload_file = gr.File(
            label="Document (.pdf, .docx, .txt)",
            file_count="single",
        )
        with gr.Row():
            upload_title = gr.Textbox(label="Title", placeholder="Optional")
            upload_doc_type = gr.Textbox(label="Doc type", placeholder="policy / faq / manual")
        with gr.Row():
            upload_uploader = gr.Textbox(label="Uploader ID", placeholder="your@email.com")
            upload_replace = gr.Checkbox(label="Replace if doc_id already exists", value=True)
        upload_btn = gr.Button("⬆️ Upload & Ingest", variant="primary")
        upload_status = gr.Markdown("")
        upload_metrics = gr.Dataframe(
            headers=["Metric", "Value"],
            value=[],
            interactive=False,
            row_count=0,
        )

    with gr.Tab("Browse & Delete"):
        refresh_docs_btn = gr.Button("🔄 Refresh documents", size="sm")
        docs_status = gr.Markdown("")
        docs_table = gr.Dataframe(
            headers=["doc_id", "title", "doc_type", "uploader", "chunks", "last_seen"],
            value=[],
            interactive=False,
            wrap=True,
            row_count=0,
        )
        gr.Markdown("**Chunks for selected document:**")
        chunks_table = gr.Dataframe(
            headers=["chunk_id", "page", "snippet"],
            value=[],
            interactive=False,
            wrap=True,
            row_count=0,
        )
        chunks_status = gr.Markdown("")

        gr.Markdown("**Delete a document:**")
        with gr.Row():
            delete_doc_id = gr.Textbox(
                label="Document ID to delete",
                placeholder="Paste doc_id from the table above",
            )
            delete_btn = gr.Button("🗑 Delete", variant="stop")
        delete_status = gr.Markdown("")

    # Wire
    refresh_domain_btn.click(
        fn=lambda: gr.update(choices=get_domain_list()),
        outputs=[domain_dd],
    )
    upload_btn.click(
        fn=on_upload_document,
        inputs=[domain_dd, upload_file, upload_title, upload_doc_type,
                upload_uploader, upload_replace],
        outputs=[upload_status, upload_metrics],
    )
    refresh_docs_btn.click(
        fn=on_refresh_documents,
        inputs=[domain_dd],
        outputs=[docs_status, docs_table],
    )
    docs_table.select(
        fn=on_select_document,
        inputs=[docs_table, domain_dd],
        outputs=[delete_doc_id, chunks_table, chunks_status],
    )
    delete_btn.click(
        fn=on_delete_document,
        inputs=[domain_dd, delete_doc_id],
        outputs=[delete_status, docs_table],
    )


# ===========================================================================
# TAB 4 — PLAYGROUND (RAG config tuner)
# ===========================================================================

def _pg_save_config(
    config_name, config_desc, vectorstore, distance_metric, collection_name,
    persist_dir, chunking_strategy, chunk_size, overlap, similarity_threshold,
    max_chunk_size, embedding_provider, embedding_model, device, batch_size,
    retrieval_strategies, top_k, hybrid_alpha, llm_provider, llm_model,
    temperature, max_tokens, session_id,
):
    today_tag = datetime.now().strftime("%d%m%Y")
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
            "hybrid": {"alpha": hybrid_alpha} if "hybrid" in retrieval_strategies else {},
        },
        "llm": {
            "provider": llm_provider,
            "model_name": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }
    PlaygroundConfigManager.save_config(full_name, today_tag, config)
    return f"✅ Config **{full_name}** saved."


def _pg_save_as_template(template_name: str, config_name: str, session_id: str):
    if not template_name:
        return "⚠️ Enter a template name."
    all_configs = PlaygroundConfigManager.list_configs()
    match = next(
        (c for c in all_configs
         if c.get("name") == config_name or c.get("playground_name") == config_name),
        None,
    )
    if not match:
        return f"❌ Config '{config_name}' not found."
    cfg = PlaygroundConfigManager.load_config(match["filename"])
    path = Path("configs/templates") / f"{template_name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml as _yaml
    with open(path, "w") as f:
        _yaml.dump(cfg, f)
    return f"⭐ Template **{template_name}** saved."


def _pg_run_query(query_text: str, config_name: str, session_id: str):
    """Run a test query using a playground config."""
    if not query_text:
        return "—", [], ""
    if not config_name:
        return "⚠️ Load a config first.", [], ""
    try:
        from core.playground_config_manager import PlaygroundConfigManager as PCM
        pg_filename = PCM.find_config_by_name(config_name)
        if not pg_filename:
            return f"❌ Config '{config_name}' not found.", [], ""
        pg_cfg = PCM.load_config(pg_filename)
        pg_mgr = PlaygroundConfigManager()
        merged = pg_mgr.merge_with_global(pg_cfg)
        synth_id = pg_cfg.get("playground_name") or config_name
        merged.setdefault("domain_id", synth_id)
        merged.setdefault("name", synth_id)

        from core.config_manager import DomainConfig
        domain_cfg = DomainConfig(**merged)

        temp_name = f"{synth_id}_playground_temp"
        temp_file = Path("configs/domains") / f"{temp_name}.yaml"
        if not temp_file.exists():
            import yaml as _yaml
            d = domain_cfg.model_dump() if hasattr(domain_cfg, "model_dump") else domain_cfg.dict()
            with open(temp_file, "w") as f:
                _yaml.safe_dump(d, f)

        svc = DocumentService(temp_name)
        result = svc.query_with_answer(query_text=query_text)
        answer = result["answer"]
        sources = result["sources"]
        rows = [[s.get("doc_id"), s.get("score"), s.get("snippet", "")[:200]] for s in sources]
        return answer, rows, f"✅ Strategy: {result['trace'].get('strategy')}"
    except Exception as e:
        logger.exception(e)
        return f"❌ {e}", [], ""


def build_tab_playground() -> None:
    vectorstore_providers = ComponentRegistry.get_vectorstore_providers()
    distance_metrics = ComponentRegistry.get_distance_metrics()
    chunking_strategies = ComponentRegistry.get_chunking_strategies()
    embedding_providers = ComponentRegistry.get_embedding_providers()
    device_options = ComponentRegistry.get_device_options()
    retrieval_strategies_options = ComponentRegistry.get_retrieval_strategies()
    llm_providers = ComponentRegistry.get_llm_providers()

    default_vs = vectorstore_providers[0] if vectorstore_providers else None
    default_dm = distance_metrics[0] if distance_metrics else None
    default_cs = chunking_strategies[0] if chunking_strategies else "recursive"
    provider_choices = list(embedding_providers.keys())
    default_ep = provider_choices[0] if provider_choices else None
    default_em_list = embedding_providers.get(default_ep, [])
    default_em = default_em_list[0] if default_em_list else None
    default_dev = device_options[0] if device_options else "cpu"
    default_rs = [retrieval_strategies_options[0]] if retrieval_strategies_options else []
    llm_provider_choices = list(llm_providers.keys())
    default_lp = llm_provider_choices[0] if llm_provider_choices else "gemini"
    default_lm_list = llm_providers.get(default_lp, [])
    default_lm = default_lm_list[0] if default_lm_list else None

    raw_session_id = str(uuid.uuid4())[:8]
    session_id = gr.State(raw_session_id)

    gr.Markdown(f"### RAG Playground &nbsp;&nbsp; `session: {raw_session_id}`")
    gr.Markdown(
        "Configure and test a RAG pipeline without creating a domain. "
        "Save good configs as templates to reuse them in domain creation."
    )

    with gr.Row():
        config_selector = gr.Dropdown(
            choices=[c["name"] for c in PlaygroundConfigManager.list_configs()],
            label="Saved configs",
            value=None,
            scale=3,
        )
        load_btn = gr.Button("📂 Load", scale=1)
        save_btn = gr.Button("💾 Save config", scale=1)

    pg_status = gr.Markdown("")

    with gr.Row():
        # LEFT: config
        with gr.Column(scale=1):
            gr.Markdown("#### Configuration")
            pg_config_name = gr.Textbox(label="Config name", placeholder="MyConfig_v1")
            pg_config_desc = gr.Textbox(label="Description", lines=2)

            with gr.Tab("Vector Store"):
                pg_vs = gr.Dropdown(vectorstore_providers, label="Provider", value=default_vs)
                pg_dm = gr.Dropdown(distance_metrics, label="Distance metric", value=default_dm)
                pg_col = gr.Textbox(label="Collection name")
                pg_persist = gr.Textbox(label="Persist directory")

            with gr.Tab("Chunking"):
                pg_cs = gr.Dropdown(chunking_strategies, label="Strategy", value=default_cs)
                pg_chunk_size = gr.Slider(100, 2000, value=500, step=50, label="Chunk size")
                pg_overlap = gr.Slider(0, 200, value=50, step=10, label="Overlap")
                pg_sim_thresh = gr.Slider(0.0, 1.0, value=0.7, step=0.01,
                                          label="Similarity threshold", visible=False)
                pg_max_chunk = gr.Slider(500, 3000, value=1000, step=50,
                                         label="Max chunk size", visible=False)

            with gr.Tab("Embeddings"):
                pg_ep = gr.Dropdown(provider_choices, label="Provider", value=default_ep)
                pg_em = gr.Dropdown(default_em_list, label="Model", value=default_em)
                pg_dev = gr.Dropdown(device_options, label="Device", value=default_dev)
                pg_batch = gr.Slider(1, 256, value=32, step=1, label="Batch size")

            with gr.Tab("Retrieval"):
                pg_rs = gr.CheckboxGroup(
                    retrieval_strategies_options, label="Strategies", value=default_rs
                )
                pg_topk = gr.Slider(1, 50, value=10, step=1, label="Top K")
                pg_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Hybrid α")

            with gr.Tab("LLM"):
                pg_lp = gr.Dropdown(llm_provider_choices, label="Provider", value=default_lp)
                pg_lm = gr.Dropdown(default_lm_list, label="Model", value=default_lm)
                pg_temp = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="Temperature")
                pg_maxtok = gr.Slider(128, 4096, value=512, step=64, label="Max tokens")

        # RIGHT: test
        with gr.Column(scale=1):
            gr.Markdown("#### Test Query")
            pg_query = gr.Textbox(label="Question", placeholder="Ask something...", lines=3)
            pg_run_btn = gr.Button("▶ Run", variant="primary")
            pg_answer = gr.Markdown(label="Answer")
            pg_chunks_df = gr.Dataframe(
                headers=["doc_id", "score", "snippet"],
                interactive=False,
                row_count=0,
                label="Retrieved chunks",
            )
            pg_run_status = gr.Markdown("")

    # Save as template
    with gr.Row():
        pg_tpl_name = gr.Textbox(
            label="Save as template — name",
            placeholder="Default_HR_Template",
            scale=3,
        )
        pg_save_tpl_btn = gr.Button("⭐ Save as template", scale=1)

    # --- Wire events ---
    def _update_emb_models(p):
        m = ComponentRegistry.get_embedding_providers().get(p, [])
        return gr.update(choices=m, value=m[0] if m else None)

    def _update_llm_models(p):
        m = ComponentRegistry.get_llm_providers().get(p, [])
        return gr.update(choices=m, value=m[0] if m else None)

    def _update_chunk_visibility(s):
        if s == "semantic":
            return gr.update(visible=False), gr.update(visible=False), \
                   gr.update(visible=True), gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=True), \
               gr.update(visible=False), gr.update(visible=False)

    pg_ep.change(_update_emb_models, inputs=[pg_ep], outputs=[pg_em])
    pg_lp.change(_update_llm_models, inputs=[pg_lp], outputs=[pg_lm])
    pg_cs.change(
        _update_chunk_visibility,
        inputs=[pg_cs],
        outputs=[pg_chunk_size, pg_overlap, pg_sim_thresh, pg_max_chunk],
    )

    save_btn.click(
        fn=_pg_save_config,
        inputs=[
            pg_config_name, pg_config_desc, pg_vs, pg_dm, pg_col, pg_persist,
            pg_cs, pg_chunk_size, pg_overlap, pg_sim_thresh, pg_max_chunk,
            pg_ep, pg_em, pg_dev, pg_batch,
            pg_rs, pg_topk, pg_alpha,
            pg_lp, pg_lm, pg_temp, pg_maxtok,
            session_id,
        ],
        outputs=[pg_status],
    )

    pg_run_btn.click(
        fn=_pg_run_query,
        inputs=[pg_query, pg_config_name, session_id],
        outputs=[pg_answer, pg_chunks_df, pg_run_status],
    )

    pg_save_tpl_btn.click(
        fn=_pg_save_as_template,
        inputs=[pg_tpl_name, pg_config_name, session_id],
        outputs=[pg_status],
    )


# ===========================================================================
# MAIN APP
# ===========================================================================

def build_admin_app() -> gr.Blocks:
    with gr.Blocks(
        title="RAG Admin Console",
        theme=gr.themes.Soft(primary_hue="slate", secondary_hue="blue"),
    ) as demo:
        gr.Markdown("# RAG Platform — Admin Console")
        gr.Markdown(
            "Manage templates, domains, and documents. "
            "User chat interface is in **app.py** (port 7860)."
        )

        with gr.Tabs():
            with gr.Tab("📋 Templates"):
                build_tab_templates()

            with gr.Tab("🗂 Domains"):
                build_tab_domains()

            with gr.Tab("📄 Documents"):
                build_tab_documents()

            with gr.Tab("🧪 Playground"):
                build_tab_playground()

    return demo


if __name__ == "__main__":
    app = build_admin_app()
    app.launch(server_name="127.0.0.1", server_port=7861, show_error=True)
