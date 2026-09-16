# AI-Knowledge-Assistant-RAG-LUNORSOFT-PVT-LTD

LUNORSOFT - INTERNSHIP RECRUITMENT - ROUND 1 - 2026 

This project is for internship role recruitment task for the startup LUNORSOFT TECHNOLOGIES PRIVATE LIMITED based https://lunor.online/ focused on AI assisted coding and learning based tools and technology. 

Note : Im not officially affiliated with the mentioned company and this is just for the task given for a recruitment process for learning purposes and assessment.

## Choosing OPTION 1 : RAG Assistant  Chatbot answering questions based on provided documents.  Python, LangChain/LlamaIndex, Vector Store (FAISS), Streamlit.  Repo, UI Demo, README.


# -------------------------------------
### mentioned things :

Option 1 — Build a Mini AI Knowledge Assistant 
Core Requirement: RAG 

Objective 
Build a simple AI-powered application that can answer questions based on a provided 
document or collection of documents. 
Requirements 

The application should: 
1.  Use a PDF/document or collection of documents as the knowledge source. 
2.  Extract and process the document content. 
3.  Split the content into appropriate chunks. 
4.  Generate embeddings for the content. 
5.  Store and retrieve relevant information using a vector store or suitable retrieval 
mechanism. 
6.  Accept questions from the user. 
7.  Retrieve relevant information and generate answers using an LLM. 
8.  Ensure that answers are primarily based on the provided knowledge source. 
9.  Provide a simple and usable interface. 

Suggested Technologies 
You may use technologies such as: 
●  Python 
●  LangChain / LlamaIndex 
●  OpenAI / Gemini / Hugging Face 
●  FAISS / Chroma / Pinecone 
●  Streamlit / Gradio 

Optional / Bonus Features 
●  Source citations 
●  Conversation history 
●  Multiple documents 
●  Improved retrieval or evaluation 
●  Deployment 

Submission Requirements 
Submit: 
●  GitHub repository 
●  Working demo/link, if available 
●  README explaining your approach and implementation 

# -------------------------------------


# All my reasoning behind the task is written below : 

Im listing out all the ground rules and requirements here so i make sure i dont miss anything 

### Why im choosing OPTION 1 :

Option 1: RAG Assistant  Chatbot answering questions based on provided documents.  Python, LangChain/LlamaIndex, Vector Store (FAISS), Streamlit.  Repo, UI Demo, README.  

Option 2: Fine-Tuning  Coding assistant trained on a custom dataset.  LoRA/QLoRA, open-source LLM, custom dataset prep.  

i have 2 options here , as i have explored lunorsoft website , and its about using the power of AI such as LLMs and more human like TTS for voice generation , and other things to generate educational and learning content , but the email also mentioned company focusses on assisted coding so i believe they are also planning to work on assisted coding tools which im very fond of as well. 

The problem here with the option 2 is that while fine tuning a model you can only fine tune it upto certain extent to leave it still useful and generalised and not overfit on the existing knowledge , theres alot we cant feed it direct so we need to make sure llm stays intact for it to still fill those gaps that is the whole point of using LLMs for assisted learning it can fill generalised gaps and be super customised to your own tailored uses , and the problem again with option 2 is that you will have a trained model for only a certain topic and youll have to retrain it if you want it to focus on the a new topic or slightly connected topic, and the maxima we get after which the llms started to get worse with finetuning and not better is called catastrophic overtraining , and personally i have seen it in Image diffusion models. and thats is why im going with Option 1, becuase i think RAG is much more of a better option to generalise the help with any document we throw at it. and it will parse it and store it in the vector embedding and we can retrieve it based on the ongoign context using something like cosine similarity or any mathematical algorithmn this fixes the inherent issue in LLMs the biggest problems in LLMs which is context memory is limited and we need to make sure we fit only the most important things as per the ongoing conversation.

In early 2023 companies were prefering finetuned llms for specific custom uses and openAI even had a program for enterprises where they can contact them to build them their custom finetuned models. but now later on the methods such as mixture of experts and chain of thoughts really fixed these issues and the frontier models outperformed the finetuned ones. and people stopped preferring the finetuned models but then its is still useful for small distilled models for mass usecases such as the Lunor option 2 and also is her talk about LoRA based fine tuning then its even more cheaper to train than finetuning the whole model, low rank adaptation model is basically a layer that sits on top of the base model , that helps the model to track the path better to and to the more revelent results , i have trained LoRA models in Diffusion models so i know how impactful they are truely , thats why the option is still good, but im going with option 1, also if i consider it like this then yes the frontier models also have larger context memory now and so even the RAG may not be needed for general things but yeah lol, we are here considering the right balance and efficiency of the system and keeping it fesible for business and cost of model. we can also go with a hybrid option where we do train a LoRA model at same time we also apply RAG for more efficient context use. then i think that would be the best option but to keep it simple as this is a time constrained assignment i would stay with option1. 

(note : for transparency i didnt not just ask this above information from an AI , i know this because i have closely following almost every research paper and new models and methods that was being developed by companies since 2022 dec 15 when the very first gpt 3 model was released to the public by OPENAI , i used multiple sources such as youtube channels such as two minute papers and more to stay upto date with these)

Sorry for such a long paragraph , Im just trying to show my reasoning behind why i choose this option , ill try to keep it concise.

### tools and docs i used:

https://developers.llamaindex.ai/python/framework/understanding/rag/ ,
https://docs.langchain.com/oss/python/langgraph/agentic-rag , 
https://gradio.app/ , https://github.com/gradio-app/gradio , 
https://developers.llamaindex.ai/python/framework-api-reference/storage/vector_store/faiss/ , 
