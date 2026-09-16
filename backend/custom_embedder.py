#this is a wrapper code that helps llamaindex work with onxx models 
# this code specifically is totally written by AI
# but this method and logic everything is mine.
# its basically inheriting the default support class
# and changing it to wrok with the onxx model embedder


import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from typing import Any , List

class ONNXGemmaEmbedding(BaseEmbedding):
    # LlamaIndex requires PrivateAttr for custom objects in BaseEmbedding
    _session: Any =  PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _query_prefix: str =  PrivateAttr()
    _doc_prefix: str = PrivateAttr()
    def __init__(self, model_id: str = "onnx-community/embeddinggemma-300m-ONNX" , **kwargs: Any) -> None:
        super().__init__(**kwargs)
        model_path = hf_hub_download(model_id, subfolder="onnx", filename="model.onnx" )
        hf_hub_download(model_id, subfolder="onnx", filename="model.onnx_data" )   
        self._session =  ort.InferenceSession( model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id )
        self._query_prefix =  "task: search result | query: "
        self._doc_prefix = "title: none | text: "

    @classmethod
    def class_name(cls) -> str :
        return "onnx_gemma"

    def _get_query_embedding(self, query: str) -> List[float] :
        text = self._query_prefix + query
        inputs = self._tokenizer([text], padding=True, return_tensors="np")
        _ , embeddings = self._session.run(None, inputs.data)
        return embeddings[0].tolist()

    def _get_text_embedding(self, text: str ) -> List[float] :
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        prefixed_texts = [self._doc_prefix + t for t in texts]
        inputs = self._tokenizer(prefixed_texts , padding=True , return_tensors="np")
        _ , embeddings = self._session.run(None , inputs.data)
        return embeddings.tolist()

    async def _aget_query_embedding(self , query: str) -> List[float]:
        return self._get_query_embedding(query)
    async def _aget_text_embedding(self , text: str) -> List[float]:
        return self._get_text_embedding(text)
    async def _aget_text_embeddings(self , texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)