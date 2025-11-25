import gradio as gr
import uuid
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import yaml

from core.registry.component_registry import ComponentRegistry


# =============================
# Backend functions
# (hooked to your existing managers)
# =============================

def load_config_list() -> List[str]:
    # Use your config manager here
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
    session_id,  # still passed in, but no longer used for filename
):
    from core.playground_config_manager import PlaygroundConfigManager

    # --- Build date-based suffix ---
    today_tag = datetime.now().strftime("%d%m%Y")   # e.g. "25032025"

    full_name = f"{config_name}_{today_tag}" if config_name else today_tag

    # --- Build config dict ---
    config = {
        "name": full_name,                      # store full name with date
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

    # UI status message
    return f"✅ Config **{full_name}** saved on `{today_tag}`."



def save_as_template(template_name: str, config_name: str, session_id: str):
    from core.playground_config_manager import PlaygroundConfigManager

    if not template_name:
        return "⚠️ Please enter a template name."

    # Load all configs
    all_configs = PlaygroundConfigManager.list_configs()

    # Try matching by name, playground_name OR filename (covers all cases)
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

    # Load the actual YAML config content
    cfg = PlaygroundConfigManager.load_config(match["filename"])

    # Write the template file
    path = Path("configs/templates") / f"{template_name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f)

    return f"⭐ Template **{template_name}** created from config **{config_name}**."

def run_pipeline(
    user_query: str,
    config_name: str,
    session_id: str,
):
    """
    Run the RAG pipeline with the current config and query.
    Replace this with your actual retrieval + LLM call.
    """
    if not user_query.strip():
        return "⚠️ Please enter a question to run.", [], "No query provided."

    # TODO: call your real RAG stack here
    answer = (
        f"Demo answer for: **{user_query}**\n\n"
        f"_Using config_ `{config_name or 'Current (unsaved) config'}` "
        f"_session_ `{session_id}`."
    )

    # Demo retrieved chunks
    demo_chunks = [
        {"doc_id": 1, "score": 0.87, "snippet": "Chunk 1 text snippet..."},
        {"doc_id": 2, "score": 0.82, "snippet": "Chunk 2 text snippet..."},
    ]
    logs = (
        "Demo pipeline executed.\nSteps:\n"
        "1) Embed query\n2) Retrieve chunks\n3) Generate answer"
    )

    return answer, demo_chunks, logs


# =============================
# UI / Playground Layout
# =============================

def build_playground() -> gr.Blocks:
    # ---- Registry-driven options & defaults ----
    vectorstore_providers = ComponentRegistry.get_vectorstore_providers()
    distance_metrics = ComponentRegistry.get_distance_metrics()
    chunking_strategies = ComponentRegistry.get_chunking_strategies()
    embedding_providers = ComponentRegistry.get_embedding_providers()  # dict: provider -> [models]
    device_options = ComponentRegistry.get_device_options()
    retrieval_strategies_options = ComponentRegistry.get_retrieval_strategies()

    default_vectorstore = vectorstore_providers[0] if vectorstore_providers else None
    default_distance_metric = distance_metrics[0] if distance_metrics else None
    default_chunking_strategy = (
        chunking_strategies[0] if chunking_strategies else "fixed"
    )

    provider_choices = list(embedding_providers.keys())
    default_provider = provider_choices[0] if provider_choices else None
    default_models = embedding_providers.get(default_provider, []) if default_provider else []
    default_model = default_models[0] if default_models else None

    llm_providers = ComponentRegistry.get_llm_providers()  # dict
    llm_provider_choices = list(llm_providers.keys())
    default_llm_provider = llm_provider_choices[0] if llm_provider_choices else None
    default_llm_models = llm_providers.get(default_llm_provider, []) if default_llm_provider else []
    default_llm_model = default_llm_models[0] if default_llm_models else None

    default_device = device_options[0] if device_options else "cpu"
    default_retrieval_strategies = (
        [retrieval_strategies_options[0]] if retrieval_strategies_options else []
    )

    with gr.Blocks(title="RAG Playground - MVP") as demo:
        # ---- Session state (avoid passing raw strings as inputs) ----
        raw_session_id = str(uuid.uuid4())[:8]
        session_id = gr.State(raw_session_id)

        # -----------------------------------------------------------------
        # Header bar: title + one-line description + config actions
        # -----------------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## 🔧 RAG Playground")
                gr.Markdown(
                    "Configure, save, and test your retrieval pipeline. "
                    "Designed to be usable even without training."
                )
                gr.Markdown(f"**Session ID:** `{raw_session_id}`")
            with gr.Column(scale=2):
                with gr.Row():
                    # Dropdown to select config when loading
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

        # Status / feedback area (visible under header)
        config_status = gr.Markdown("ℹ️ No config actions yet.")

        # -----------------------------------------------------------------
        # Main area: LEFT = configuration tabs, RIGHT = test/debug tabs
        # -----------------------------------------------------------------
        with gr.Row():
            # ---------------- LEFT: Configuration ----------------
            with gr.Column(scale=3):
                # Shared config name & description at top
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

                    # Show/hide depending on strategy
                    chunk_size = gr.Slider(
                        100,
                        2000,
                        value=500,
                        step=50,
                        label="Chunk Size",
                        visible=True,
                    )
                    overlap = gr.Slider(
                        0,
                        200,
                        value=50,
                        step=10,
                        label="Overlap",
                        visible=True,
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
                            1,
                            256,
                            value=32,
                            step=1,
                            label="Batch size",
                        )

                # ---- Tab 4: Retrieval ----
                with gr.Tab("4. Retrieval"):
                    retrieval_strategies = gr.CheckboxGroup(
                        retrieval_strategies_options,
                        label="Retrieval strategies",
                        value=default_retrieval_strategies,
                    )
                    top_k = gr.Slider(
                        1,
                        50,
                        value=10,
                        step=1,
                        label="Top K",
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

            # ---------------- RIGHT: Playground / Testing ----------------
            with gr.Column(scale=2):
                gr.Markdown("### 🧪 Playground")

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

        # -----------------------------------------------------------------
        # Dynamic behaviors (models, chunking params)
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
            # Example logic from documentation:
            # - "recursive" & "fixed" -> chunk_size + overlap
            # - "semantic" -> similarity_threshold + max_chunk_size
            if strategy == "semantic":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )
            else:  # "fixed" or "recursive" or others
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
        # Wiring interactions: Load, Save, Save as Template, Run
        # -----------------------------------------------------------------

        # 1. Load config

        def on_load_config(selected_name: str, session_id_value: str):
            if not selected_name:
                # return same number of outputs as below
                return (
                    "⚠️ Please select a config to load.",
                    gr.update(),  # config_name
                    gr.update(),  # config_desc
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

            # Vectorstore
            vs_provider_val = vectorstore_cfg.get("provider", default_vectorstore)
            vs_distance_val = vectorstore_cfg.get("distance_metric", default_distance_metric)
            vs_collection_val = vectorstore_cfg.get("collection_name", "")
            vs_persist_val = vectorstore_cfg.get("persist_directory", "")

            # Chunking
            strategy_val = chunking_cfg.get("strategy", default_chunking_strategy)
            strategy_params = chunking_cfg.get(strategy_val, {}) or {}

            chunk_size_val = strategy_params.get("chunk_size", 500)
            overlap_val = strategy_params.get("overlap", 50)
            similarity_threshold_val = strategy_params.get("similarity_threshold", 0.7)
            max_chunk_size_val = strategy_params.get("max_chunk_size", 1000)

            # Embeddings
            emb_provider_val = embeddings_cfg.get("provider", default_provider)
            emb_model_val = embeddings_cfg.get("model_name", default_model)
            device_val = embeddings_cfg.get("device", default_device)
            batch_size_val = embeddings_cfg.get("batch_size", 32)

            # Retrieval
            retrieval_strats_val = retrieval_cfg.get(
                "strategies", default_retrieval_strategies
            )
            top_k_val = retrieval_cfg.get("top_k", 10)
            hybrid_cfg = retrieval_cfg.get("hybrid", {}) or {}
            hybrid_alpha_val = hybrid_cfg.get("alpha", 0.5)

            # 👇 NOTE: second = config_name, third = description
            return (
                status,
                cfg.get("name", selected_name),  # config_name
                cfg.get("description", ""),  # config_desc
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

        def on_load_config(selected_name: str, session_id_value: str):
            if not selected_name:
                # return same number of outputs as below
                return (
                    "⚠️ Please select a config to load.",
                    gr.update(),  # config_name
                    gr.update(),  # config_desc
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

            # Vectorstore
            vs_provider_val = vectorstore_cfg.get("provider", default_vectorstore)
            vs_distance_val = vectorstore_cfg.get("distance_metric", default_distance_metric)
            vs_collection_val = vectorstore_cfg.get("collection_name", "")
            vs_persist_val = vectorstore_cfg.get("persist_directory", "")

            # Chunking
            strategy_val = chunking_cfg.get("strategy", default_chunking_strategy)
            strategy_params = chunking_cfg.get(strategy_val, {}) or {}

            chunk_size_val = strategy_params.get("chunk_size", 500)
            overlap_val = strategy_params.get("overlap", 50)
            similarity_threshold_val = strategy_params.get("similarity_threshold", 0.7)
            max_chunk_size_val = strategy_params.get("max_chunk_size", 1000)

            # Embeddings
            emb_provider_val = embeddings_cfg.get("provider", default_provider)
            emb_model_val = embeddings_cfg.get("model_name", default_model)
            device_val = embeddings_cfg.get("device", default_device)
            batch_size_val = embeddings_cfg.get("batch_size", 32)

            # Retrieval
            retrieval_strats_val = retrieval_cfg.get(
                "strategies", default_retrieval_strategies
            )
            top_k_val = retrieval_cfg.get("top_k", 10)
            hybrid_cfg = retrieval_cfg.get("hybrid", {}) or {}
            hybrid_alpha_val = hybrid_cfg.get("alpha", 0.5)

            # 👇 NOTE: second = config_name, third = description
            return (
                status,
                cfg.get("name", selected_name),  # config_name
                cfg.get("description", ""),  # config_desc
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
                config_name,  # 👈 now updated from loaded config
                config_desc,
                vectorstore,
                distance_metric,
                collection_name,
                persist_dir,
                chunking_strategy,
                chunk_size,
                similarity_threshold,
                overlap,
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
        # 2. Save config
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
                session_id,  # ✅ gr.State, not raw string
            ],
            outputs=config_status,
        )

        # 3. Save as template
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

        # 4. Run pipeline
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
            # Convert list[dict] -> rows for dataframe
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

    return demo


if __name__ == "__main__":
    app = build_playground()
    app.launch()
