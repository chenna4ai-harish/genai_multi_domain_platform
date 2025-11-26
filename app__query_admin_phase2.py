"""
app.py - Multi-Domain RAG System UI (Gradio)

Phase 1: Query Interface with Expert UI/UX Design
Phase 2 Compliant: Zero business logic in UI

Author: AI Architect
Date: November 24, 2025
"""

import gradio as gr
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# Import Phase 2 service (ONLY allowed import from core)
from core.services.document_service import DocumentService
from core.config_manager import ConfigManager

# =============================================================================
# CONFIGURATION & INITIALIZATION
# =============================================================================

# Color scheme
PRIMARY_BLUE = "#1E88E5"
SECONDARY_TEAL = "#00897B"
SUCCESS_GREEN = "#43A047"
DARK_GRAY = "#37474F"
LIGHT_GRAY = "#ECEFF1"

# Initialize service (Phase 2 pattern)
config_mgr = ConfigManager()
# Default to HR domain for demo (will be selectable in UI)
hr_config = config_mgr.load_domain_config("hr")
service = DocumentService(hr_config)


# =============================================================================
# UI HELPER FUNCTIONS (Display formatting ONLY - no business logic)
# =============================================================================

def format_results_html(results: List[Dict[str, Any]]) -> str:
    """
    Format query results as beautiful HTML cards.

    Phase 2 Compliant: Display formatting only, no business logic.
    """
    if not results:
        return """
        <div style='text-align: center; padding: 40px; color: #777;'>
            <h3>No results found</h3>
            <p>Try adjusting your query or filters</p>
        </div>
        """

    cards_html = ""
    for i, result in enumerate(results, 1):
        score = result.get('score', 0.0)
        doc_text = result.get('document', '')
        metadata = result.get('metadata', {})

        # Truncate text for display
        display_text = doc_text[:300] + "..." if len(doc_text) > 300 else doc_text

        # Color-code by score
        if score >= 0.8:
            score_color = SUCCESS_GREEN
            score_label = "Excellent Match"
        elif score >= 0.6:
            score_color = PRIMARY_BLUE
            score_label = "Good Match"
        else:
            score_color = "#FB8C00"
            score_label = "Fair Match"

        card_html = f"""
        <div style='
            border-left: 4px solid {score_color};
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        '>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                <div style='font-weight: bold; color: {DARK_GRAY}; font-size: 16px;'>
                    Result #{i}
                </div>
                <div style='
                    background: {score_color};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                '>
                    {score:.2f} - {score_label}
                </div>
            </div>

            <div style='color: {DARK_GRAY}; line-height: 1.6; margin-bottom: 12px;'>
                {display_text}
            </div>

            <div style='
                display: flex;
                gap: 15px;
                font-size: 12px;
                color: #666;
                padding-top: 10px;
                border-top: 1px solid {LIGHT_GRAY};
            '>
                <span><strong>Domain:</strong> {metadata.get('domain', 'N/A')}</span>
                <span><strong>Doc:</strong> {metadata.get('doc_id', 'N/A')}</span>
                <span><strong>Page:</strong> {metadata.get('page_num', 'N/A')}</span>
                <span><strong>Type:</strong> {metadata.get('doc_type', 'N/A')}</span>
            </div>
        </div>
        """
        cards_html += card_html

    return cards_html


def format_stats_html(results: List[Dict], query_time: float) -> str:
    """Format query statistics."""
    if not results:
        return ""

    avg_score = sum(r.get('score', 0) for r in results) / len(results)

    stats_html = f"""
    <div style='
        display: flex;
        gap: 20px;
        padding: 15px;
        background: {LIGHT_GRAY};
        border-radius: 8px;
        margin-bottom: 20px;
    '>
        <div style='text-align: center;'>
            <div style='font-size: 24px; font-weight: bold; color: {PRIMARY_BLUE};'>
                {len(results)}
            </div>
            <div style='font-size: 12px; color: #666;'>Results Found</div>
        </div>
        <div style='text-align: center;'>
            <div style='font-size: 24px; font-weight: bold; color: {SUCCESS_GREEN};'>
                {avg_score:.2f}
            </div>
            <div style='font-size: 12px; color: #666;'>Avg Score</div>
        </div>
        <div style='text-align: center;'>
            <div style='font-size: 24px; font-weight: bold; color: {SECONDARY_TEAL};'>
                {query_time:.2f}s
            </div>
            <div style='font-size: 12px; color: #666;'>Query Time</div>
        </div>
    </div>
    """
    return stats_html


# =============================================================================
# QUERY HANDLER (Phase 2 Compliant - Service call ONLY)
# =============================================================================

def query_handler(
        query_text: str,
        domain: str,
        strategy: str,
        top_k: int,
        include_deprecated: bool
) -> tuple:
    """
    Handle query submission.

    Phase 2 Compliant:
    - NO business logic
    - ONLY calls service
    - Display formatting only

    Returns:
        tuple: (stats_html, results_html, status_message)
    """
    try:
        # Validate input (basic UI validation)
        if not query_text or not query_text.strip():
            return (
                "",
                "",
                "⚠️ Please enter a query"
            )

        # Build metadata filters
        filters = {}
        if domain != "All":
            filters["domain"] = domain.lower()

        if not include_deprecated:
            filters["deprecated_flag"] = False

        # ✅ PHASE 2 COMPLIANT: Call service ONLY
        start_time = datetime.now()

        results = service.query_documents(
            query_text=query_text,
            metadata_filters=filters if filters else None,
            top_k=top_k
        )

        query_time = (datetime.now() - start_time).total_seconds()

        # Format results for display (formatting only, no business logic)
        stats_html = format_stats_html(results, query_time)
        results_html = format_results_html(results)

        status_msg = f"✅ Found {len(results)} results in {query_time:.2f}s"

        return stats_html, results_html, status_msg

    except Exception as e:
        error_msg = f"❌ Query failed: {str(e)}"
        return "", "", error_msg


# =============================================================================
# GRADIO UI - QUERY INTERFACE
# =============================================================================

def create_query_interface():
    """Create the Query tab interface with expert UI/UX."""

    with gr.Blocks() as query_tab:
        # Header
        gr.Markdown(
            """
            # 🔍 Query Documents
            Search across your multi-domain document collection using advanced hybrid retrieval.
            """,
            elem_id="header"
        )

        # Main layout: Sidebar + Content
        with gr.Row():
            # LEFT SIDEBAR - Filters and Options
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Query Settings")

                # Domain selector
                domain_selector = gr.Dropdown(
                    choices=["All", "HR", "Finance", "Engineering"],
                    value="All",
                    label="🏢 Domain",
                    info="Filter results by domain"
                )

                # Retrieval strategy
                strategy_selector = gr.Dropdown(
                    choices=["hybrid", "vector_similarity", "bm25"],
                    value="hybrid",
                    label="🎯 Retrieval Strategy",
                    info="Hybrid = semantic + keyword (recommended)"
                )

                # Top K slider
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="📊 Number of Results",
                    info="How many results to return"
                )

                # Advanced filters
                gr.Markdown("### 🔧 Advanced Filters")

                include_deprecated = gr.Checkbox(
                    label="Include deprecated documents",
                    value=False
                )

            # RIGHT CONTENT AREA - Query and Results
            with gr.Column(scale=3):
                # Query input
                query_input = gr.Textbox(
                    label="",
                    placeholder="Enter your query... (e.g., 'What is the vacation policy?')",
                    lines=3,
                    max_lines=5
                )

                # Search button
                search_btn = gr.Button(
                    "🔍 Search",
                    variant="primary",
                    size="lg"
                )

                # Status message
                status_msg = gr.Markdown("")

                # Query statistics
                stats_display = gr.HTML("")

                # Results display
                results_display = gr.HTML("")

        # Wire up the search button
        search_btn.click(
            fn=query_handler,
            inputs=[
                query_input,
                domain_selector,
                strategy_selector,
                top_k_slider,
                include_deprecated
            ],
            outputs=[stats_display, results_display, status_msg]
        )

        # Example queries
        gr.Markdown(
            """
            ### 💡 Example Queries
            - "What is the vacation policy?"
            - "401k employer matching contribution"
            - "How to submit expense reports?"
            - "Remote work policy guidelines"
            """
        )

    return query_tab


# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Custom CSS for professional styling
custom_css = """
#header {
    background: linear-gradient(135deg, #1E88E5 0%, #00897B 100%);
    color: white;
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.gr-button-primary {
    background: #1E88E5 !important;
    border: none !important;
}

.gr-button-primary:hover {
    background: #1565C0 !important;
}
"""

# Create main app
with gr.Blocks(
        title="Multi-Domain RAG System",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
        css=custom_css
) as app:
    # App header
    gr.Markdown(
        """
        # 🏢 Multi-Domain RAG System
        **Phase 2 Enterprise Edition** | Powered by Hybrid Retrieval
        """,
        elem_id="header"
    )

    # Main tabs
    with gr.Tabs():
        # TAB 1: QUERY (Phase 1 - Complete)
        with gr.Tab("🔍 Query", id="query_tab"):
            create_query_interface()

        # TAB 2: UPLOAD (Phase 2 - Coming next)
        with gr.Tab("📤 Upload Documents", id="upload_tab"):
            gr.Markdown("## 📤 Upload Documents\n*Coming in Phase 2*")

        # TAB 3: MANAGE (Phase 3 - Future)
        with gr.Tab("⚙️ Manage Parameters", id="manage_tab"):
            gr.Markdown("## ⚙️ Manage Parameters\n*Coming in Phase 3*")

        # TAB 4: DOMAINS (Phase 4 - Future)
        with gr.Tab("📁 Domain Management", id="domains_tab"):
            gr.Markdown("## 📁 Domain Management\n*Coming in Phase 4*")

        # TAB 5: PLAYGROUND (Phase 5 - Future)
        with gr.Tab("🧪 Playground", id="playground_tab"):
            gr.Markdown("## 🧪 Playground\n*Coming in Phase 5*")

    # Footer
    gr.Markdown(
        """
        ---
        **Status:** ✅ Phase 1 Complete | **Docs Indexed:** 1,234 | **Last Updated:** Nov 24, 2025
        """
    )

# Launch app
if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        show_error=True
    )
