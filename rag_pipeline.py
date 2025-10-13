import os
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- ✅ Authenticate Hugging Face properly ---
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Successfully logged in to Hugging Face using HF_TOKEN.")
    except Exception as e:
        print(f"⚠️ Hugging Face token login failed: {e}")
else:
    print("⚠️ No HF_TOKEN found in .env file. Using public mode.")

def build_rag_pipeline():
    """
    Builds the complete RAG pipeline and returns:
    - llm: the reasoning LLM for general queries
    - retriever: vector retriever
    - qa_chain: RAG QA chain
    """

    # --- Load dataset ---
    try:
        dataset = load_dataset("daily_dialog", split="train[:300]")
        print("✅ Loaded public dataset: daily_dialog")
    except Exception as e:
        print(f"⚠️ Could not load 'daily_dialog': {e}")
        dataset = load_dataset("OpenRL/daily_dialog", split="train[:300]", token=HF_TOKEN)

    # --- Convert dialogs to strings and split into chunks ---
    texts = [" ".join(d["dialog"]) for d in dataset]
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=t) for t in texts]
    split_docs = splitter.split_documents(docs)

    # --- Embeddings + Chroma DB ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma_db")

    # --- Retriever ---
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # --- LLM for reasoning / fallback ---
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)

    # --- RAG QA chain ---
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return llm, retriever, qa_chain
