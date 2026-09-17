
import os
from dotenv import load_dotenv
from llama_index.core import Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai_like import OpenAILike
from custom_embedder import ONNXGemmaEmbedding
from indexer import load_precomputed_index, process_and_index

load_dotenv()
# global embedder
Settings.embed_model = ONNXGemmaEmbedding()

# since im using free api here ,we cant be sure when any one of them dies so 
# im gonna use all of the 3 as a backup to eachother if one falls down
# ill start the another one with the fallback logic 
modelPriority = [
    #"google/gemma-4-31b-it:free",
    "liquid/lfm-2.5-2.6b:free" # only one working 
    #"google/gemma-4-26b-a4b-it:free"
]


mainPrompt = """
You are an AI agent who's task is to teach to the user 
the user is a student , and you have to the task to act as the
knowledge distributor who dosent make up data rather grounps it from the 
given embedding if not given then dont make up data unless you are 100% sure
if you have no valid data then tell the user directly about it.

work efficiently , teach cleanly and write your explanations in
proper formatting , dont assume anything about the user that he knows what
be very clear and specific to even the smallest details thanks.

Context information is below:
---------------------
{context_str}
---------------------
Given this information, answer the user query: {query_str}
"""

qaPrompt = PromptTemplate(mainPrompt)

class KnowledgeAgent:
    def __init__(self, base_storage_dir: str = "storage"):
        self.base_storage_dir = base_storage_dir
        self.mode = None
        self.active_data = None
        self.data = None
        self.llms = []
        for m in modelPriority:
            self.llms.append(OpenAILike(
                model=m,
                api_base="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                is_chat_model=True,
                additional_kwargs={"extra_body": {"reasoning": {"enabled": True}}}
            ))

    def load_books(self, book_names: list[str]):
        if not book_names:
            self.mode = None
            self.data = None
            return True, "No documents selected. Switched to General Chat mode."
            
        loaded_indices = []
        for book in book_names:
            folder = os.path.join(self.base_storage_dir, book)
            if os.path.exists(folder):
                try:
                    vstore = FaissVectorStore.from_persist_dir(persist_dir=folder)
                    ctx = StorageContext.from_defaults(vector_store=vstore, persist_dir=folder)
                    index = load_index_from_storage(storage_context=ctx)
                    loaded_indices.append(index)
                except Exception as e:
                    print(f"Failed to load {book}: {e}")
                
        if not loaded_indices:
            return False, "Failed to load the selected vector indices."
            
        self.data = loaded_indices
        self.mode = "index"
        return True, f"Successfully loaded {len(loaded_indices)} document(s) for RAG!"

    def load_user_files(self, files: list[str]):
        if not files:
            self.mode = None
            self.data = None
            return True, "Upload cleared. Switched to General Chat mode."            
        res = process_and_index(files)
        if res["mode"] == "empty":
            return False, "Failed to parse files or files empty."
        self.mode = res["mode"]
        self.data = res["data"] if self.mode == "direct" else [res["data"]]       
        if self.mode == "direct":
            return True, "Loaded document directly (Bypassed embedding)."
        return True, "Generated live vector index."
    
    def ask(self, query: str):
        err = ""
        for llm in self.llms:
            try:
                if not self.mode or self.data is None:
                    res = llm.complete(query)
                    return str(res), [], llm.model
                elif self.mode == "direct":
                    final_prompt = qaPrompt.format(
                        context_str=self.data,
                        query_str=query
                    )
                    res = llm.complete(final_prompt)         
                    refs = [{
                        "id": 1,
                        "file": "Full Document",
                        "page": "All (Direct Context)",
                        "score": "Direct Injection",
                        "snippet": self.data[:200].replace("\n", " ")
                    }]
                    return str(res), refs, llm.model
                elif self.mode == "index":
                    all_nodes = []
                    for data_index in self.data:
                        retriever = data_index.as_retriever(similarity_top_k=2)
                        nodes = retriever.retrieve(query)
                        all_nodes.extend(nodes)
                    context_str = "\n\n".join([n.node.get_text() for n in all_nodes])
                    final_prompt = qaPrompt.format(context_str=context_str, query_str=query)                
                    res = llm.complete(final_prompt)  
                    refs = []
                    for rank, node in enumerate(all_nodes, start=1):
                        meta = node.metadata
                        refs.append({
                            "id": rank,
                            "file": meta.get("file_name", "Unknown"),
                            "page": meta.get("page_label", "N/A"),
                            "score": f"{node.score:.4f}" if node.score is not None else "N/A",
                            "snippet": node.node.get_text().strip().replace("\n", " ")[:150]
                        })
                    return str(res), refs, llm.model
            except Exception as e:
                print(f"Fallback triggered: {llm.model} failed. Error: {e}")
                err = str(e)
                continue
        return f"All models failed. Last error: {err}", [], "Failed"