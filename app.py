import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
import gradio as gr
from backend.agents import KnowledgeAgent
import spaces

STORAGE_DIR =  os.path.join(BASE_DIR, "storage")            
agent  = KnowledgeAgent( base_storage_dir=STORAGE_DIR)


@spaces.GPU(duration=1)
def dummy_gpu():
    return None

def get_preloaded_books():      
    if not os.path.exists(STORAGE_DIR):
        return []
    return [d for d in  os.listdir(STORAGE_DIR) if os.path.isdir(os.path.join(STORAGE_DIR, d))]

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

def handle_ask(query, chat_history):
    if not query.strip():
        return chat_history, "", "No citations.", "None"  
    ans, refs, model = agent.ask(query)  
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": ans})
    ref_md = ""
    for r in refs:
        ref_md += (
            f"**[{r['id']}] {r['file']}** — *Page {r['page']}* "
            f"(Score: `{r['score']}`)\n\n"
            f"> \"{r['snippet']}...\"\n\n---\n"
        )   
    if not ref_md:
        ref_md = "No citations retrieved. (General Chat mode or unticked checkbox)"       
    return chat_history, "", ref_md, f"Active API: {model}"

def clear_chat():
    msg = agent.clear_history()
    return [], "", "No citations.", msg

with gr.Blocks(title="Lunorsoft Knowledge Assistant") as demo:
    gr.Markdown("# AI Knowledge Assistant - Ishan Mani Singh")
    gr.Markdown("this is running based on free api by openrouter , and rag is built on top of llamaindex vector embedding FAISS")
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
                label= "Upload Live Documents (PDF/Images)", 
                file_count = "multiple"
            )            
            status = gr.Textbox(label ="System Status", interactive=False , value="Ready. No documents loaded (General Chat Mode).")            
            checkboxes.change(fn=handle_books , inputs=checkboxes, outputs=status)
            upload.change( fn=handle_upload , inputs=upload, outputs=status)
            
        with gr.Column(scale=2) :
            gr.Markdown( "### 2. Chat Agent (With Memory)")
            chatbot = gr.Chatbot(label="Conversation History", height=300)           
            with gr.Row() :
                query_box = gr.Textbox(label="Your Question", placeholder="Ask about the documents...", scale=4)
                ask_btn = gr.Button("Submit", variant="primary" , scale=1)
                clear_btn = gr.Button("Clear Chat", variant="stop" , scale=1)
            
            model_status = gr.Textbox(label="Fallback Status Tracker" , interactive=False)            
            with gr.Accordion("View Source Citations", open=False) :
                cit_box = gr.Markdown()            
            ask_btn.click(
                fn=handle_ask, 
                inputs=[query_box, chatbot], 
                outputs=[chatbot , query_box, cit_box, model_status]
            )            
            clear_btn.click(
                fn=clear_chat ,
                inputs=[] ,
                outputs=[chatbot, query_box, cit_box, model_status]
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft() )