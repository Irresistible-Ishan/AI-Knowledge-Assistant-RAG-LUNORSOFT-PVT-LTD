# using this code we will make a vector embedding maker using the parsed code we made , FAISS

import os
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from parser import loadDocs, should_bypass_embedding, extract_combined_text
from custom_embedder import ONNXGemmaEmbedding
from dotenv import load_dotenv

Settings.embed_model = ONNXGemmaEmbedding()
Settings.node_parser = SentenceSplitter( chunk_size=512 , chunk_overlap = 50)

load_dotenv()

def process_and_index(file_paths : list[str] , persist_dir : str = "storage/temp_upload"):
    docs = loadDocs(file_paths)
    if not docs:
        return {"mode" : "empty" , "data" : None }
    if should_bypass_embedding(docs):
        direct_text = extract_combined_text(docs)
        return {"mode": "direct", "data": direct_text}
    vectorStore = FaissVectorStore(faiss_index = faiss.IndexFlatL2(768))
    storageCTX = StorageContext.from_defaults(vector_store=vectorStore)
    index = VectorStoreIndex.from_documents(docs, 
        storage_context = storageCTX , 
        show_progress = True
    )
    os.makedirs(persist_dir , exist_ok= True)

    index.storage_context.persist( persist_dir = persist_dir)
    return {"mode": "index" , "data": index}

def load_precomputed_index(book_name: str, base_storage: str = "storage"):
    persist_dir = os.path.join(base_storage, book_name)
    if not os.path.exists(persist_dir):
        return None
    vectorStore = FaissVectorStore.from_persist_dir(persist_dir =persist_dir)
    storageCTX = StorageContext.from_defaults(
        vector_store=vectorStore, 
        persist_dir=persist_dir
    )
    return load_index_from_storage(storage_context = storageCTX )

if __name__ == "__main__":
    test_file = ["preloaded_docs/2312.02519v2.pdf"]
    if os.path.exists(test_file[0]):
        result = process_and_index(test_file)
        print( f"Indexing complete. Mode: {result['mode']}" )


