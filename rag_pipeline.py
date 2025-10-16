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
        dataset = load_dataset("jkhedri/psychology-dataset", split="train[:300]")
        print("✅ Loaded dataset: jkhedri/psychology-dataset")
    except Exception as e:
        print(f"⚠️ Could not load dataset: {e}")
        dataset = load_dataset("psychology-dataset", split="train[:300]", token=HF_TOKEN)

    # --- Inspect dataset structure ---
    print("📊 --- DATASET INFO ---")
    print(f"Number of samples: {len(dataset)}")
    print(f"Column names: {dataset.column_names}")
    print("📘 --- SAMPLE ENTRIES ---")
    for i in range(3):
        print(f"🧩 Row {i}:")
        for col in dataset.column_names:
            print(f"  {col}: {dataset[i][col]}")
        print()

    # --- Convert dataset into Document objects ---
    # Use question + helpful response (response_j) as text
    texts = [f"Q: {d['question']}\nA: {d['response_j']}" for d in dataset if 'response_j' in d and d['response_j'].strip()]
    
    if not texts:
        raise ValueError("No valid text found in dataset to create embeddings!")

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=t) for t in texts]
    split_docs = splitter.split_documents(docs)

    # --- Embeddings + Chroma DB ---
    #HuggingFaceEmbeddings(...)
        #Loads the pretrained embedding model (all-MiniLM-L6-v2).
        #It’s ready to convert any text → embedding vector.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #Chroma.from_documents(...)
    #Takes the text chunks (split_docs) and runs the embedding model on each one.
        #This creates a vector for each chunk.
        #Then, it stores those vectors in a vector database (Chroma DB).
    vector_db = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma_db")
    #So:
        #The embedding model creates the vectors (embeddings).
        #The vector database stores them, so later the retriever can quickly find which chunks are closest (most semantically similar) to the user’s question.


    # --- Retriever ---
    #Finds the most relevant chunks (closest vectors)
    #The retriever uses the same embedding model to convert user's question (prompt) into a vector (e.g. retriever.get_relevant_documents(query) in app.py).
    #It compares that question vector to all the stored document vectors in the database.
    #Using similarity search (like cosine similarity), it finds the top k=3 most similar chunks.
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # --- LLM for reasoning / fallback ---

    # Initialize the language model client (Gemini). This creates the LLM object,
    # which can be used directly for free-form answers (e.g. llm.invoke(query) in app.py)
    # and is also used by the RAG chain below to generate answers using retrieved context.
    #It “thinks” and generates a coherent answer. This is the brain that actually reads the retrieved text and formulates a natural-language answer.
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)

    # --- RAG QA chain ---
    # Create a RetrievalQA chain that ties the retriever and the LLM together. We are calling the retriever to get the top relevant documents for the "query" send in the code "qa_chain.run(query)" in app.py, call the LLM to generate the final, context-aware answer.
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return llm, retriever, qa_chain
