import os
from dotenv import load_dotenv
from llama_index.core import Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from custom_embedder import ONNXGemmaEmbedding

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
You are an AI agent who's task is to talk to the user and 
"""
