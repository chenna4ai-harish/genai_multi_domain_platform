import gradio as gr
import uuid
from pathlib import Path
import yaml

from core.registry.component_registry import ComponentRegistry
from core.playground_config_manager import PlaygroundConfigManager

# Util
def get_recent_configs():
    configs = PlaygroundConfigManager.list_configs()
    configs_sorted = sorted(configs, key=lambda x: x["created_at"], reverse=True)
    return ([f'{c["name"]} (Last edited {c["created_at"][:19]})' for c in configs_sorted], [c["filename"] for c in configs_sorted])

def update_embedding_models(provider):
    return gr.Dropdown(choices=ComponentRegistry.get_embedding_providers()[provider], value=None)

def update_chunking_params(strategy):
    return (
        gr.update(visible=strategy == "recursive"),
        gr.update(visible=strategy == "recursive"),
        gr.update(visible=strategy == "semantic"),
        gr.update(visible=strategy == "semantic")
    )

def save_config_wizard(name, desc, vec, dist, col, dir_, strategy, c_sz, ov, s_thr, mc_sz, emb_prov, emb_mod, dev, bs, ret_strats, tk, alpha, session_id):
    if not name or not all([vec, dist, strategy, emb_prov, emb_mod, ret_strats]):
        return "❌ Please fill all required fields.", gr.update(visible=False), False, False
    config = {
        "name": name,
        "description": desc,
        "vectorstore": {'provider': vec, 'distance_metric': dist, 'collection_name': col, 'persist_directory': dir_},
        "chunking": {'strategy':strategy, strategy: {'chunk_size':c_sz, 'overlap':ov, 'similarity_threshold':s_thr, 'max_chunk_size':mc_sz }},
        "embeddings":{'provider':emb_prov, 'model_name':emb_mod, 'device':dev, 'batch_size':bs},
        "retrieval":{'strategies':ret_strats, 'top_k':tk, 'hybrid':{'alpha':alpha} if "hybrid" in ret_strats else {}}
    }
    yaml_path = PlaygroundConfigManager.save_config(name, session_id, config)
    return f"✅ Config saved to {yaml_path}", gr.update(visible=True), True, False

def load_config_fields(filename):
    cfg = PlaygroundConfigManager.load_config(filename)
    chunking_recursive = cfg.get("chunking",{}).get("recursive",{})
    chunking_semantic = cfg.get("chunking",{}).get("semantic",{})
    return [
        cfg.get("name",""), cfg.get("description",""), cfg.get("vectorstore",{}).get("provider"),
        cfg.get("vectorstore",{}).get("distance_metric"), cfg.get("vectorstore",{}).get("collection_name"),
        cfg.get("vectorstore",{}).get("persist_directory"), cfg.get("chunking",{}).get("strategy"),
        chunking_recursive.get("chunk_size",500), chunking_recursive.get("overlap",50),
        chunking_semantic.get("similarity_threshold",0.7), chunking_semantic.get("max_chunk_size",1000),
        cfg.get("embeddings",{}).get("provider","sentencetransformers"),
        cfg.get("embeddings",{}).get("model_name","all-MiniLM-L6-v2"),
        cfg.get("embeddings",{}).get("device","cpu"),
        cfg.get("embeddings",{}).get("batch_size",32),
        cfg.get("retrieval",{}).get("strategies",["hybrid"]),
        cfg.get("retrieval",{}).get("top_k",10),
        cfg.get("retrieval",{}).get("hybrid",{}).get("alpha",0.7),
        True, True
    ]

def advance_to_step(step_idx, *args):
    # Only make the selected step visible (set all others to False)
    return [gr.update(visible=(i == step_idx)) for i in range(4)]

def save_as_template(template_name, loaded_cfg_filename,session_id):
    cfg = PlaygroundConfigManager.load_config(loaded_cfg_filename)
    if not cfg: return "❌ Config not loaded or found."
    path = Path("configs/templates") / f"{template_name}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return f"✅ Saved as template: {path}"

def delete_config(filename):
    ok, msg = PlaygroundConfigManager.delete_config(filename)
    return msg

with gr.Blocks(title="Playground: Wizard UI") as playground:
    raw_session_id = str(uuid.uuid4())[:8]
    session_id = gr.State(raw_session_id)

    session_display = gr.Markdown(f"**Session ID:** `{raw_session_id}`", elem_id="session-id-banner")
    step_labels = ["Choose or Start", "Configure", "Test/Experiment", "Export/Manage"]

    visible_steps = [gr.State(i==0) for i in range(4)]
    current_config_filename = gr.State("")

    with gr.Row():
        gr.Markdown("> **What do you want to do?** <br>Start new experiment or continue/edit a previous one.")

    # Step 0: Choose Experiment
    with gr.Group(visible=visible_steps[0]) as step0_select:
        with gr.Row():
            start_new_btn = gr.Button("📝 Start New Experiment", elem_id="start-new-btn")
            recent_names, recent_filenames = get_recent_configs()
            continue_recent = gr.Dropdown(choices=recent_names, label="Continue Recent Experiment") if recent_names else gr.Markdown("No recent experiments.")
    session_display = gr.Markdown(f"**Session ID:** `{session_id}`", elem_id="session-id-banner")
    step0_next_status = gr.Markdown()

    # Step 1: Configure (Form)
    with gr.Group(visible=visible_steps[1]) as step1_configure:
        gr.Markdown("### Step 1 of 4: Configure Your Experiment")
        with gr.Row():
            config_name = gr.Textbox(label="Experiment Name *", placeholder="e.g. exp_test_HR_2025")
            config_desc = gr.Textbox(label="Description", placeholder="Short description for you.")
        with gr.Row():
            vectorstore = gr.Dropdown(ComponentRegistry.get_vectorstore_providers(), label="Vector Store Provider")
            distance_metric = gr.Dropdown(ComponentRegistry.get_distance_metrics(), label="Distance Metric")
            collection_name = gr.Textbox(label="Collection Name")
            persist_dir = gr.Textbox(label="Persist Directory")
        with gr.Row():
            chunking_strategy = gr.Dropdown(ComponentRegistry.get_chunking_strategies(), label="Chunking Strategy", value="recursive")
            chunk_size = gr.Slider(100, 2000, value=500, label="Chunk Size", visible=True)
            overlap = gr.Slider(0, 200, value=50, label="Overlap", visible=True)
            similarity_threshold = gr.Slider(0.0, 1.0, value=0.7, label="Similarity Threshold", visible=False)
            max_chunk_size = gr.Slider(500, 3000, value=1000, label="Max Chunk Size", visible=False)
        with gr.Row():
            embedding_provider = gr.Dropdown(list(ComponentRegistry.get_embedding_providers().keys()), label="Embedding Provider", value="sentencetransformers")
            embedding_model = gr.Dropdown(ComponentRegistry.get_embedding_providers()["sentencetransformers"], label="Embedding Model")
            device = gr.Dropdown(ComponentRegistry.get_device_options(), label="Device", value="cpu")
            batch_size = gr.Slider(8, 128, value=32, label="Batch Size")
        with gr.Row():
            retrieval_strategies = gr.CheckboxGroup(ComponentRegistry.get_retrieval_strategies(), label="Retrieval Strategies", value=["hybrid"])
            top_k = gr.Slider(1, 50, value=10, label="Top K Results")
            hybrid_alpha = gr.Slider(0.0, 1.0, value=0.7, label="Hybrid Alpha")
        config_status = gr.Markdown()
        save_btn = gr.Button("💾 Save & Continue")
        to_step2_btn = gr.Button("→ Next: Test", visible=False)
        loaded_flag = gr.State(False)
        save_flag = gr.State(False)

    # Step 2: Test/Experiment
    with gr.Group(visible=visible_steps[2]) as step2_test:
        gr.Markdown("### Step 2 of 4: Experiment/Test [Pipeline logic here]")
        back1 = gr.Button("← Back")
        forward2 = gr.Button("→ Next: Export")

    # Step 3: Export & Manage
    with gr.Group(visible=visible_steps[3]) as step3_export:
        gr.Markdown("### Step 3 of 4: Export & Template Management")
        with gr.Row():
            template_name_for_save = gr.Textbox(label="Export As Template Name")
            save_as_template_btn = gr.Button("⎙ Export as Template")
        exp_status = gr.Markdown()
        back2 = gr.Button("← Back")
        gr.Markdown("#### Delete configs:")
        cfg_files = get_recent_configs()[1]
        del_cfg_dropdown = gr.Dropdown(cfg_files, label="Delete Playground Config")
        del_btn = gr.Button("🧹 Delete Config")
        del_status = gr.Markdown()

    # -- Navigation & actions --
    def go_to_config_form():
        return [False, True, False, False]  # Only step 1

    def go_to_test():
        return [False, False, True, False]

    def go_to_export():
        return [False, False, False, True]

    def go_back_to_config():
        return [False, True, False, False]

    def go_back_to_test():
        return [False, False, True, False]

    start_new_btn.click(go_to_config_form, None, visible_steps)
    save_btn.click(
        save_config_wizard,
        inputs=[config_name, config_desc, vectorstore, distance_metric, collection_name, persist_dir,
                chunking_strategy, chunk_size, overlap, similarity_threshold, max_chunk_size, embedding_provider,
                embedding_model, device, batch_size, retrieval_strategies, top_k, hybrid_alpha, session_id],
        outputs=[config_status, to_step2_btn, save_flag, loaded_flag]
    )
    to_step2_btn.click(go_to_test, None, visible_steps)
    back1.click(go_back_to_config, None, visible_steps)
    forward2.click(go_to_export, None, visible_steps)
    back2.click(go_back_to_test, None, visible_steps)
    del_btn.click(delete_config, del_cfg_dropdown, del_status)
    embedding_provider.change(update_embedding_models, embedding_provider, embedding_model)
    chunking_strategy.change(update_chunking_params, chunking_strategy, [chunk_size, overlap, similarity_threshold, max_chunk_size])
    save_as_template_btn.click(
        save_as_template,
        inputs=[template_name_for_save, config_name],  # Use experiment name for config file
        outputs=[exp_status]
    )
    if recent_names:
        def on_continue_recent(selected):
            idx = recent_names.index(selected)
            fname = recent_filenames[idx]
            flds = load_config_fields(fname)
            return [False, True, False, False] + flds
        continue_recent.change(
            on_continue_recent, continue_recent,
            visible_steps + [
                config_name, config_desc, vectorstore, distance_metric, collection_name, persist_dir,
                chunking_strategy, chunk_size, overlap, similarity_threshold, max_chunk_size, embedding_provider,
                embedding_model, device, batch_size, retrieval_strategies, top_k, hybrid_alpha, loaded_flag, save_flag
            ]
        )

playground.launch()
