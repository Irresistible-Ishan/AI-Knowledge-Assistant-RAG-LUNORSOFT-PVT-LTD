import os
import sys

# 1. Lock the absolute path of the project folder dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Inject backend path safely
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

import gradio as gr
from backend.agents import KnowledgeAgent

# 3. Force the absolute path for storage so it ALWAYS finds your folders
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
agent = KnowledgeAgent(base_storage_dir=STORAGE_DIR)

def get_preloaded_books():
    # It will now always look precisely inside AI-Knowledge-Assistant.../storage/
    if not os.path.exists(STORAGE_DIR):
        return []
    return [d for d in os.listdir(STORAGE_DIR) if os.path.isdir(os.path.join(STORAGE_DIR, d))]

def handle_books(book_names):
    success, msg = agent.load_books(book_names)
    return msg

def handle_upload(files):
    if not files:
        success, msg = agent.load_user_files([])
        return msg
    
    file_paths = [file.name for file in files]
    success, msg = agent.load_user_files(file_paths)
    return msg

def handle_ask(query):
    if not query.strip():
        return "Please enter a question.", "No citations.", "None"
    
    ans, refs, model = agent.ask(query)
    
    ref_md = ""
    for r in refs:
        ref_md += (
            f"**[{r['id']}] {r['file']}** — *Page {r['page']}* "
            f"(Score: `{r['score']}`)\n\n"
            f"> \"{r['snippet']}...\"\n\n---\n"
        )
    
    if not ref_md:
        ref_md = "No citations retrieved. (Direct injection or General Chat mode)"
        
    return ans, ref_md, f"Active API: {model}"

with gr.Blocks(title="Lunorsoft Knowledge Assistant") as demo:
    gr.Markdown("# Lunorsoft AI Knowledge Assistant")
    gr.Markdown("Zero-cost, CPU-optimized RAG implementation featuring dynamic ONNX embeddings and automated API failovers.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Data Source")
            
            checkboxes = gr.CheckboxGroup(
                choices=get_preloaded_books(), 
                label="Select Pre-computed Indexes", 
                info="Select multiple books. Leave unchecked for general AI chat."
            )
            
            gr.Markdown("*Note: <= 10 pages bypasses embedding. > 10 pages triggers ONNX indexing.*")
            upload = gr.File(
                label="Upload Live Documents (PDF/Images)", 
                file_count="multiple"
            )
            
            status = gr.Textbox(label="System Status", interactive=False, value="Ready. No documents loaded (General Chat Mode).")
            
            checkboxes.change(fn=handle_books, inputs=checkboxes, outputs=status)
            upload.change(fn=handle_upload, inputs=upload, outputs=status)
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Chat Agent")
            query_box = gr.Textbox(label="Your Question", placeholder="Ask about the documents, or ask a general question...")
            ask_btn = gr.Button("Submit Query", variant="primary")
            
            ans_box = gr.Textbox(label="Generated Answer", lines=6, interactive=False)
            model_status = gr.Textbox(label="Fallback Status Tracker", interactive=False)
            
            with gr.Accordion("View Source Citations", open=False):
                cit_box = gr.Markdown()
            
            ask_btn.click(
                fn=handle_ask, 
                inputs=query_box, 
                outputs=[ans_box, cit_box, model_status]
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())