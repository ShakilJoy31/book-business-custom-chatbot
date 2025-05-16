import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG Chatbot API")

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Define request model
class QueryRequest(BaseModel):
    query: str
    
# Define response model
class QueryResponse(BaseModel):
    result: str

# Initialize vectorstore and LLM only once when the API starts
embedding_model = None
vectorstore = None
qa_chain = None


def initialize_components():
    global embedding_model, vectorstore, qa_chain
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Load vector store
    try:
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vectorstore: {str(e)}")
        raise e
    
    # Set up custom prompt
    custom_prompt_template = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context. 

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    
    # Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"max_length": 512},
        huggingfacehub_api_token='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'
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not qa_chain:
        raise HTTPException(status_code=500, detail="System not initialized properly")
    
    try:
        response = qa_chain.invoke({'query': request.query})
        return {"result": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Add a health check endpoint
@app.get("/")
async def health_check():
    if vectorstore and qa_chain:
        return {"status": "Chatbot is running successfully."}
    return {"status": "unhealthy"}

if __name__ == "__main__":
    import uvicorn
    initialize_components()
    uvicorn.run(app, host="0.0.0.0", port=8000)