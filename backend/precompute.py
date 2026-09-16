import os
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from parser import loadDocs
from custom_embedder import ONNXGemmaEmbedding

Settings.embed_model = ONNXGemmaEmbedding()
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def precompute_books(source_dir: str = "preloaded_docs", base_storage: str = "storage"):
    if not os.path.exists(source_dir):
        print(f"dir {source_dir} not found")
        return
    for filename in os.listdir(source_dir):
        if not filename.endswith(".pdf"):
            continue       
        filepath = os.path.join(source_dir, filename)
        persist_dir = os.path.join(base_storage, os.path.splitext(filename)[0])
        if os.path.exists(persist_dir):
            print(f"{os.path.splitext(filename)[0]} already exists")
            continue
        print(f"\nprecomputing embedding using ONNX index for: {os.path.splitext(filename)[0]}...")
        docs = loadDocs([filepath])
        
        if docs:
            # Gemma 300 M uses 768 dimensions thats why ill use the sam ehere
            faiss_index = faiss.IndexFlatL2(768)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex.from_documents(
                docs, storage_context=storage_context, show_progress=True
            )
            
            os.makedirs(persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)
            print(f"Success! Saved to {persist_dir}")

if __name__ == "__main__":
    precompute_books()