import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import httpx
import json

# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()

app = FastAPI(title="Facebook Chatbot API")

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")
FACEBOOK_PAGE_ACCESS_TOKEN = os.environ.get("FACEBOOK_PAGE_ACCESS_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Define request models
class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    result: str

class FacebookMessage(BaseModel):
    text: str

class FacebookSender(BaseModel):
    id: str

class FacebookMessaging(BaseModel):
    sender: FacebookSender
    message: FacebookMessage
    postback: Optional[Dict] = None

class FacebookEntry(BaseModel):
    messaging: List[FacebookMessaging]

class FacebookWebhook(BaseModel):
    object: str
    entry: List[FacebookEntry]

# Initialize components
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
        huggingfacehub_api_token=HF_TOKEN
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )

async def get_medibot_response(user_message: str) -> str:
    """Get response from custom chatbot"""
    try:
        if not qa_chain:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")
            
        response = qa_chain.invoke({'query': user_message})
        return response["result"]
    except Exception as e:
        print(f"Error getting Medibot response: {str(e)}")
        raise e

async def get_deepseek_response(user_message: str) -> str:
    """Fallback to DeepSeek when custom chatbot doesn't know the answer"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Chatbot",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek/deepseek-r1:free",
                    "messages": [{"role": "user", "content": user_message}]
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"DeepSeek API error: {response.text}"
                )
                
            data = response.json()
            return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error getting DeepSeek response: {str(e)}")
        raise e

async def send_facebook_message(recipient_id: str, message_text: str):
    """Send message back to Facebook user"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://graph.facebook.com/v12.0/me/messages?access_token={FACEBOOK_PAGE_ACCESS_TOKEN}",
                json={
                    "recipient": {"id": recipient_id},
                    "message": {"text": message_text}
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Facebook API error: {response.text}"
                )
                
            print(f"Message sent to {recipient_id}")
    except Exception as e:
        print(f"Error sending Facebook message: {str(e)}")
        raise e

async def handle_facebook_message(sender_id: str, message_text: str):
    """Process incoming Facebook message"""
    try:
        # First try custom chatbot
        response = await get_medibot_response(message_text)
        
        # If the response indicates it doesn't know, fall back to DeepSeek
        if "don't know" in response.lower() or "dont know" in response.lower():
            deepseek_response = await get_deepseek_response(message_text)
            await send_facebook_message(sender_id, deepseek_response)
        else:
            await send_facebook_message(sender_id, response)
    except Exception as e:
        error_message = "Sorry, I'm having trouble responding right now. Please try again later."
        await send_facebook_message(sender_id, error_message)
        print(f"Error handling message: {str(e)}")

@app.post("/webhook")
async def facebook_webhook(request: Request):
    """Handle incoming Facebook webhooks"""
    # Handle verification
    if request.method == "GET":
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        
        if mode and token:
            if mode == "subscribe" and token == FACEBOOK_PAGE_ACCESS_TOKEN:
                print("WEBHOOK_VERIFIED")
                return challenge
            return {"status": "verification_failed"}, 403
        return {"status": "bad_request"}, 400
    
    # Handle incoming messages
    body = await request.json()
    print("Received webhook:", body)
    
    if body.get("object") != "page":
        return {"status": "not_found"}, 404
    
    # Process each entry
    for entry in body.get("entry", []):
        for messaging in entry.get("messaging", []):
            if messaging.get("message"):
                sender_id = messaging["sender"]["id"]
                message_text = messaging["message"]["text"]
                await handle_facebook_message(sender_id, message_text)
            elif messaging.get("postback"):
                print("Received postback:", messaging["postback"])
                # Handle postbacks if needed
    
    return {"status": "event_received"}, 200

@app.get("/")
async def health_check():
    """Health check endpoint"""
    if vectorstore and qa_chain:
        return {"status": "Chatbot is running successfully"}
    return {"status": "unhealthy"}, 500

if __name__ == "__main__":
    import uvicorn
    initialize_components()
    uvicorn.run(app, host="0.0.0.0", port=8000)























































# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Dict, Any, Optional

# # LangChain imports
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint

# # Load environment variables
# load_dotenv()

# app = FastAPI(title="RAG Chatbot API")

# # Constants
# DB_FAISS_PATH = "vectorstore/db_faiss"
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# HF_TOKEN = os.environ.get("HF_TOKEN")



# # Define request model
# class QueryRequest(BaseModel):
#     query: str
    
# # Define response model
# class QueryResponse(BaseModel):
#     result: str

# # Initialize vectorstore and LLM only once when the API starts
# embedding_model = None
# vectorstore = None
# qa_chain = None


# def initialize_components():
#     global embedding_model, vectorstore, qa_chain
    
#     # Initialize embedding model
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
#     # Load vector store
#     try:
#         vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     except Exception as e:
#         print(f"Error loading vectorstore: {str(e)}")
#         raise e
    
#     # Set up custom prompt
#     custom_prompt_template = """
#     Use the pieces of information provided in the context to answer user's question.
#     If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#     Dont provide anything out of the given context. 

#     Context: {context}
#     Question: {question}

#     Start the answer directly. No small talk please.
#     """
    
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    
#     # Initialize LLM
#     llm = HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         task="text-generation",
#         temperature=0.5,
#         model_kwargs={"max_length": 512},
#         huggingfacehub_api_token=HF_TOKEN
#     )
    
#     # Create QA chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#         return_source_documents=False,
#         chain_type_kwargs={'prompt': prompt}
#     )

# @app.post("/query", response_model=QueryResponse)
# async def process_query(request: QueryRequest):
#     print(f"HF_TOKEN: {HF_TOKEN}")
#     print("Hello world.")
#     if not qa_chain:
#         raise HTTPException(status_code=500, detail="System not initialized properly")
    
#     try:
#         response = qa_chain.invoke({'query': request.query})
#         return {"result": response["result"]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# # Add a health check endpoint
# @app.get("/")
# async def health_check():
#     if vectorstore and qa_chain:
#         return {"status": "Chatbot is running successfully."}
#     return {"status": "unhealthy"}

# if __name__ == "__main__":
#     import uvicorn
#     initialize_components()
#     uvicorn.run(app, host="0.0.0.0", port=8000)