# using this code we will make a vector embedding maker using the parsed code we made , FAISS

import os
import faiss
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
# https://developers.llamaindex.ai/python/framework-api-reference/storage/vector_store/faiss/
from parser import loadDocs 

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)




