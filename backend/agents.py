
import os
from dotenv import load_dotenv
from llama_index.core import Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from custom_embedder import ONNXGemmaEmbedding
from indexer import load_precomputed_index, process_and_index

load_dotenv()
# global embedder
Settings.embed_model = ONNXGemmaEmbedding()

# since im using free api here ,we cant be sure when any one of them dies so 
# im gonna use all of the 3 as a backup to eachother if one falls down
# ill start the another one with the fallback logic 
modelPriority = [
    "google/gemma-4-31b-it:free",
    "liquid/lfm-2.5-2.6b:free",
    "google/gemma-4-26b-a4b-it:free"
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
        self.llms = []
        for m in modelPriority:
            self.llms.append(OpenAI(
                model=m,
                api_base="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                additional_kwargs={"extra_body": {"reasoning": {"enabled": True}}}
            ))

    def load_book(self, book_name: str):
        persist_dir = os.path.join(self.base_storage_dir, book_name)
        if not os.path.exists(persist_dir):
            return False , f"Could not find index for {book_name} in {persist_dir}."
            
        try:
            vectorStore = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
            storageCTX = StorageContext.from_defaults(
                vector_store = vectorStore , persist_dir=persist_dir
            )
            self.active_data = load_index_from_storage(storage_context = storageCTX)
            self.mode = "index"
            return True , f"loaded {book_name}! "
        except Exception as e:
            return False , f"Failed to load vector: {str(e)} "

    def load_user_files( self , file_paths: list[str] ):
        result = process_and_index(file_paths )
        if result["mode"] == "empty" :
            return False, "Failed to parse files or files empty. "
        self.mode = result["mode"]
        self.active_data = result["data"] 
        if self.mode == "direct" :
            return True , "Loaded document directly (Bypassed embedding)."
        return True , "Generated live vector index."
    
    def ask(self, query: str) :
        if not self.mode or self.active_data is None :
            return "Error: No active document loaded." , [] , "None"
        last_error  = ""
        for llm in self.llms :
            try:
                # this mode is direct for no embedding if context is small enough
                if self.mode == "direct" :
                    formatted_prompt = qaPrompt.format(
                        context_str=self.active_data,
                        query_str=query
                    )
                    response = llm.complete(formatted_prompt)
                    citations = [{
                        "id": 1,
                        "file": "Full Document",
                        "page": "All (Direct Context)",
                        "score": "Direct Injection",
                        "snippet": self.active_data[:200].replace( "\n", " ")
                    }]
                    return str(response), citations, llm.model

                # 2nd mode for embedding based context call 
                # took help of ai here for docs and retrieval
                elif self.mode == "index" :
                    query_engine = self.active_data.as_query_engine(
                        llm=llm,
                        similarity_top_k=3,
                        text_qa_template=qaPrompt
                    )
                    response = query_engine.query(query)
                    citations = []
                    if hasattr(response, "source_nodes"):
                        for rank, node in enumerate(response.source_nodes, start=1):
                            meta = node.metadata
                            citations.append({
                                "id": rank,
                                "file": meta.get("file_name", "Unknown"),
                                "page": meta.get("page_label", "N/A"),
                                "score": f"{node.score:.4f}" if node.score is not None else "N/A",
                                "snippet": node.node.get_text().strip().replace("\n", " ")[:200]
                            })
                    return str(response) , citations , llm.model

            except Exception as e:
                print(f"Fallback triggered : {llm.model} failed. Error : {e}")
                last_error = str(e)
                continue

        return f"All models failed. Last error: {last_error}" , [] , "Failed"