import asyncio
import os 
import pandas as pd
import hashlib
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Body
from pydantic import BaseModel
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_qdrant import QdrantVectorStore 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from fastapi.responses import JSONResponse

from uuid import uuid4

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


input_folder = "Car_Details"
output_folder = "Extracted_Car_Details"
collection_name = "car_details"

# Load all environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("GROQ_API_KEY not found in .env file. Please set it.")

poppler_path = os.getenv("poppler_path")
if not poppler_path:
    print("poppler_path not found in .env file. Please set it.")
    exit(1)

pytesseract.pytesseract.tesseract_cmd = os.getenv("tesseract_cmd")
if not pytesseract.pytesseract.tesseract_cmd:
    print("tesseract_cmd not found in .env file. Please set it.")
    exit(1)

llm_model = os.getenv("llm_model")
if not llm_model:
    print("llm_model not found in .env file. Please set it.")
    exit(1)

embedding_model = os.getenv("embedding_model")
if not embedding_model:
    print("embedding_model not found in .env file. Please set it.")
    exit(1)

qdrant_url = os.getenv("qdrant_url")
print(qdrant_url,"qdrant url")
if not qdrant_url:
    print("qdrant_url not found in .env file. Please set it.")
    exit(1)




# Hashing the file
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Store embedding data history in qdrant
def store_embedding(qdrant, vector, file_hash, filename, collection_name=collection_name):
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=file_hash,  # Use file hash as unique ID
                vector=vector,
                payload={
                    "filename": filename,
                    "file_hash": file_hash
                }
            )
        ]
    )

# Check qdrant has all embedding file
def has_embeddings_qdrant():
    # Check if the Qdrant collection has any points
    qdrant_client = QdrantClient(
        # url="http://qdrant-custom.onrender.com:6333",
        url=qdrant_url,
        prefer_grpc=False,
        https=True,
        timeout=60,
        port=None,
        check_compatibility=False
    )

    #print(qdrant_client.collection_exists("car_details"))

    try:
        result, _= qdrant_client.scroll(
            collection_name=collection_name,
            limit=1
        )
        # print(result)
        return len(result) > 0 if result else False
    except Exception as e:
        print(f"Error checking Qdrant collection: {e}")
        return False
    

# import time

# def wait_for_qdrant_ready(retries=5, delay=3):
#     for i in range(retries):
#         try:
#             client = QdrantClient(url=qdrant_url, prefer_grpc=False)
#             collections = client.get_collections()
#             print("✅ Qdrant collections found:", collections)
#             return True
#         except Exception as e:
#             print(f"⏳ Waiting for Qdrant... retry {i+1}/{retries}")
#             time.sleep(delay)
#     raise RuntimeError("❌ Qdrant is not responding.")

# wait_for_qdrant_ready()



# Collecting all the embedding file name list
def list_embedded_files(collection_name):
    client = QdrantClient(
        # host="localhost",
        # port=6333,
        url=qdrant_url,
        port=None,
        https=True,)
    result, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True
    )
    source_set = set()
    for point in result:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        source = metadata.get("source", "")
        # print("payload:", source)
        source_set.add(source)
    # print("Source set :",list(source_set))
    return list(source_set)

# Extract text from the CSV files
def extract_text_from_csv(csv_path, output_folder=output_folder):
    try:
        df = pd.read_csv(csv_path)
        # print("DataFrame loaded successfully.", df.head())
        text = df.to_string(index=False)
        
        os.makedirs(output_folder, exist_ok=True)
        # output_file = os.path.join(output_folder, os.path.basename(pdf_path).replace('.csv', '.txt'))
        output_file = os.path.join(output_folder, "data.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"Extracted text from {csv_path} and saved to {output_file}")
        return text

    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return ""



def embedding_documents(output_folder, collection_name=collection_name, embedding_model=embedding_model, qdrant_url=qdrant_url):
    embedding = HuggingFaceEmbeddings(
    model_name=embedding_model,
    )
    qdrant_client = QdrantClient(
        url=qdrant_url,
        prefer_grpc=False,
        timeout=60,
        https=True,
        port=None
    ) 

    qdrant=None
    try:
        existing_collections = qdrant_client.get_collections().collections
        print(existing_collections,"hello")
        if collection_name not in [col.name for col in existing_collections]:
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384, 
                    distance= Distance.COSINE
                )
            )
            print(f"Created new collection: {collection_name}")
        else:
            print(f"Collection {collection_name} already exists. No need to recreate it.")
            
        qdrant = QdrantVectorStore(    
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embedding,
        ) 
        print(f"Connected to Qdrant collection: {collection_name}")

    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
    print("Qdrant client initialized successfully.")

    for filename in os.listdir(output_folder):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(output_folder, filename)
            file_hash = get_file_hash(file_path)
            embedded_files = list_embedded_files(collection_name)

            if filename in [_.split('\\')[1] for _ in embedded_files]:
                print(f"Skipping {filename}, already embedded in list of Qdrant.")
                continue


            loader = TextLoader(os.path.join(output_folder, filename), encoding="utf-8") # creating text loader object
            documents = loader.load() # loading documents from the text file

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # creating text splitter object
            split_docs = text_splitter.split_documents(documents) # splitting documents into smaller chunks
            print(f"Processed {len(split_docs)} chunks from {filename}.")

            QdrantVectorStore.from_documents(
                documents=split_docs,
                embedding=embedding,
                collection_name=collection_name,
                url=qdrant_url,
                # force_recreate=True # if true then it will recreate the collection if it already exists
            )
            store_embedding(
                qdrant=qdrant_client,
                vector=embedding.embed_documents([doc.page_content for doc in split_docs]),
                file_hash=file_hash,
                filename=filename,
                collection_name=collection_name
            )
    return qdrant




def build_qa_chain(qdrant, groq_api_key=groq_api_key, llm_model=llm_model):
    llm = ChatGroq(
        model=llm_model,
        api_key=groq_api_key,
        temperature=0.1,
        # max_tokens=1000,
    )

    retriever = qdrant.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    # These all for retriever
    retriever_prompt = (
        """
            Given a chat history and the latest user question which might reference context in the chat history.
            Formulate a standlone question which can be understood without the chat history.
            Do not answer the question just reformulate it if needed and otherwise return it as is.
        """
    )
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )


    history_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)


    # These all for LLM
    system_prompt = (
            """
            You are an expert assistant helping users find cars from structured listings (CSV tables, tabular data, or structured lists).

            You are given car data as context below.  
            Your job: accurately answer the user's question by applying all filters mentioned in their query, using ONLY the context provided.

            ---
            Car Data (Context):
            {context}
            ---

            Instructions:

            Carefully analyze the user’s question for any search/filter requirements, such as:
            1. Brand (e.g., Maruti, Hyundai)
            2. Model
            3. Registration Year
            3. Price or price range (e.g., 2 to 5 lakhs) with unit
            4. Number of seats (e.g., 7 seats, 5-seater)
            5. Year of make (e.g., after 2020, between 2018 and 2023)
            6. Fuel type (petrol, diesel, electric, etc.)
            7. Number of seats (e.g., 7 seats, 5-seater)
            8. Mileage 
            9. km driven
            10. Body type (SUV, sedan, hatchback, etc.)
            11. Color
            12. Transmission (manual, automatic)
            13. Insurance
            14. Engine
            15. Power
            16. Location, or any other property present in the data

            * Apply ALL relevant filters simultaneously. For example, if the user asks for "cars with 7 seats made after 2020," only show cars matching both criteria.
            * Return ALL matching cars in paragraph and in well structured formate.
            * If the question is about a summary, statistic, or comparison (e.g., "cheapest diesel car after 2020"), answer directly from the context, showing the specific value/car(s) requested.
            * Never invent, hallucinate, or add data that is not in the context. If nothing matches, say "No matching cars found in the data."
            * If the user's question is not specific, summarize or show the most relevant information as best you can using the context.
            * Do not use external knowledge; only use what is in the context.
            Please answer the user's question as accurately and clearly as possible, using only the information in the context.
            Given a chat history and the latest user question which might reference context in the chat history.

            User Question: {input}
            
            Answer:
            """
    )

    QA_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    QA_chain = create_stuff_documents_chain(llm, QA_prompt)


    # Combine retriever and LLM
    rag_chain = create_retrieval_chain(history_retriever, QA_chain)


    return rag_chain



store: Dict[str, BaseChatMessageHistory] = {}
qa_chain = None
qdrant = None

@app.on_event("startup")
async def initialize_qa_chain():
    global qa_chain, qdrant
    if has_embeddings_qdrant():
        qdrant = embedding_documents(output_folder, collection_name)
        qa_chain = build_qa_chain(qdrant)
        print("QA Chain initialized successfully.")
    else:
        print("No embedding found. Please embed the document first.")


class QuestionRequest(BaseModel):
    session_id: str
    question: str


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@app.post("/embed")
def embed_documents():
    global qa_chain, qdrant
    extract_text_from_csv(input_folder, output_folder)
    qdrant = embedding_documents(output_folder, collection_name)
    qa_chain = build_qa_chain(qdrant)
    return {"message":"Documents embedded and QA chain initialized."}

wait_time = False

@app.post("/chat")
async def ask_question(data: QuestionRequest):

    global qa_chain, wait_time

    if wait_time:
        await asyncio.sleep(30) 
        wait_time = True

    if qa_chain is None:
        raise HTTPException(
            status_code=500,
            detail = "No embeddings found. Please embed documents first."
        )

    chain_with_history = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    result = chain_with_history.invoke(
        {"input": data.question},
        config={"configurable": {"session_id": data.session_id}}
    )

    # Save messages to backend (but don't return to user)
    # get_session_history(data.session_id).add_user_message(data.question)
    # get_session_history(data.session_id).add_ai_message(result["answer"])
    print(result["answer"])

    return {"answer": result["answer"]}
  
class SessionRequest(BaseModel):
    session_id: str

@app.post("/refresh")
async def refresh_chat(Sessiondata: SessionRequest, request: Request):
    # try:
    #     body = await request.body()
    #     if not body:
    #         raise HTTPException(status_code=400, detail="Empty request body")
    #     data = await request.json()
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    
    # session_id = data.get("session_id")

    session_id = Sessiondata.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail= "Missing session id in request")
    
    if session_id in store:
        del store[session_id]
        return {"message":f"Session {session_id} refresh successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"Session id not found")


@app.get("/get_messages")
async def get_messages(session_id:str):
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found.")
    history = store[session_id].messages
    return JSONResponse(content=[
        {"sender":"user" if "HumanMessage" in str(type(m)) else "bot", "text":m.content}
        for m in history
    ])

@app.post("/new_chat")
async def new_chat():
    new_session_id = str(uuid4())
    store[new_session_id] = ChatMessageHistory()
    return JSONResponse(content={"session_id": new_session_id})


#### uvicorn main:app --reload


# npx create-react-app my-react-app
# cd my-react-app
# npm start 

