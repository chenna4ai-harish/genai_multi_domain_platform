import gradio as gr

# Dummy data/functions for illustration
domains = ["HR", "Finance", "Engineering"]


def create_domain(new_domain):
    if new_domain and new_domain not in domains:
        domains.append(new_domain)
        return f"Domain '{new_domain}' created.", domains
    return "Domain name cannot be empty or already exists.", domains


def delete_domain(domain_to_delete):
    if domain_to_delete in domains:
        domains.remove(domain_to_delete)
        return f"Domain '{domain_to_delete}' deleted.", domains
    return "Domain not found.", domains


with gr.Blocks() as demo:
    with gr.Tab("Ask"):
        gr.Markdown("## Ask — Query & Search")
        with gr.Row():
            domain = gr.Dropdown(domains, label="Select Domain", scale=1)
            query = gr.Textbox(label="Type your question", placeholder="Ask anything...", lines=5, scale=4)
        examples = gr.Markdown("Examples: What is the leave policy?")
        answer = gr.Markdown()
        btn = gr.Button("Ask")
        btn.click(lambda d, q: f"**Result for '{q}' in {d}**", inputs=[domain, query], outputs=answer)

    with gr.Tab("Domain Management"):
        gr.Markdown("## Domain Management")
        gr.Markdown("Create new domains or delete existing ones below.")

        with gr.Row():
            new_domain = gr.Textbox(label="New Domain Name")
            create_btn = gr.Button("Create Domain")
        create_status = gr.Markdown()
        domain_list = gr.Dropdown(domains, label="Existing Domains")
        delete_btn = gr.Button("Delete Domain")
        delete_status = gr.Markdown()

        # Creation workflow
        create_btn.click(
            fn=create_domain,
            inputs=[new_domain],
            outputs=[create_status, domain_list]
        )
        # Deletion workflow
        delete_btn.click(
            fn=delete_domain,
            inputs=[domain_list],
            outputs=[delete_status, domain_list]
        )


    # --- other tabs as previously provided ...

demo.launch(server_name="127.0.0.1", server_port=7862, share=False, show_error=True)
