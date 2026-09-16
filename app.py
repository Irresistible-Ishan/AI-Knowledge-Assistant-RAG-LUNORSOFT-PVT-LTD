import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
import gradio as gr
from backend.agents import KnowledgeAgent


agent = KnowledgeAgent(base_storage_dir="storage")

def get_preloaded_books() :
    if not os.path.exists("storage"):
        return []
    return  [ d for d in os.listdir( "storage" )  if os.path.isdir(os.path.join( "storage" , d) ) ]

def handle_book(book_name):
    if not book_name:
        return "Please select a document."
    success , msg = agent.load_book(book_name)
    return msg

def handle_upload(files):
    if not files :
        return  "No files uploaded."
    file_paths = [file.name for file in files]
    success , msg = agent.load_user_files(file_paths)
    return msg

def handle_ask(query) :
    if not query.strip():
        return "Please enter a question.", "No citations.", "None"
    
    ans , refs , model = agent.ask(query)

    ref_md = ""
    for r in refs:
        ref_md += (
            f"**[{r['id']}] {r['file']}** — *Page {r['page']}* "
            f"(Score: `{r['score']}`)\n\n"
            f"> \"{r['snippet']}...\"\n\n---\n"
        )
    
    if not ref_md:
        ref_md = "No citations retrieved. (Direct injection or out-of-context query)"
    return ans, ref_md, f"Active API: {model}"

# took help of ai for gradio interface WEBUI 
with gr.Blocks(title="Knowledge Assistant") as demo:
    gr.Markdown("# AI Knowledge Assistant -- Ishan Mani Sing 25BDS1119")
    gr.Markdown("Zero-cost, CPU-optimized RAG implementation featuring dynamic ONNX embeddings and automated API failovers.")

    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### 1. Data Source")

            dropdown = gr.Dropdown(
                choices=get_preloaded_books(), 
                label="Select Pre-computed Index", 
                info="Instantly loads pre-computed FAISS vectors."
            )
            
            gr.Markdown("*Note: <= 10 pages bypasses embedding. > 10 pages triggers ONNX indexing.*")
            upload = gr.File(
                label="Upload Live Documents (PDF/Images)", 
                file_count="multiple"
            )
            
            status = gr.Textbox(label="System Status", interactive=False)

            dropdown.change(fn=handle_book, inputs=dropdown, outputs=status)
            upload.upload(fn=handle_upload, inputs=upload, outputs=status)

        with gr.Column(scale=2):
            gr.Markdown("### 2. Chat Agent")
            query_box = gr.Textbox(label="Your Question", placeholder="Ask about the loaded document...")
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
    # Gradio 6.0
    demo.launch(theme=gr.themes.Soft())