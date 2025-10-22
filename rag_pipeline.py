import os
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login
from dotenv import load_dotenv
from typing import TypedDict, List

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Authenticate Hugging Face
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Logged in to Hugging Face using HF_TOKEN.")
    except Exception as e:
        print(f"⚠️ Hugging Face login failed: {e}")
else:
    print("⚠️ No HF_TOKEN found in .env file. Using public mode.")


# --- STATE DEFINITION ---
class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    chat_history: List[str]
    source_documents: List[Document]


def build_rag_pipeline():
    """Builds a LangGraph-based RAG pipeline compatible with LangChain 1.x."""

    # --- Load dataset ---
    try:
        dataset = load_dataset("fadodr/mental_health_therapy", split="train[:300]")
        print("✅ Loaded dataset: fadodr/mental_health_therapy")
    except Exception as e:
        print(f"⚠️ Could not load dataset: {e}")
        dataset = load_dataset("mental_health_therapy", split="train[:300]", token=HF_TOKEN)

    # --- Prepare documents ---
    texts = [f"Q: {d['instruction']}\nA: {d['input']}" for d in dataset if d.get("input", "").strip()]
    if not texts:
        raise ValueError("No valid text found in dataset to create embeddings!")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=t) for t in texts]
    split_docs = splitter.split_documents(docs)

    # --- Embeddings + Chroma DB ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma_db")
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # --- LLM ---
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

    # --- PROMPT TEMPLATE ---
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. Use the following retrieved context to answer the user's question.
        If the context doesn't contain the answer, say so politely.
        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # --- NODES (GRAPH FUNCTIONS) ---
    #"""Retrieve relevant documents using vector search."""
    def retrieve_docs(state: RAGState):
        query = state["question"]
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        return {"context": context, "source_documents": docs}

    #"""Generate answer using LLM with retrieved context."""
    def generate_answer(state: RAGState):
        prompt_text = prompt.format(context=state["context"], question=state["question"])
        response = llm.invoke(prompt_text)
        return {"answer": response.content}

    # --- BUILD THE GRAPH ---
    graph_builder = StateGraph(RAGState)
    graph_builder.add_node("retrieve", retrieve_docs)
    graph_builder.add_node("generate", generate_answer)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")

    # Add in-memory checkpointing (conversation memory)
    memory = MemorySaver()

    graph = graph_builder.compile(checkpointer=memory)

    # --- Wrap in a callable interface for app.py ---
    # LangGraph graphs require a structured "state" and are not regular functions.
    # RAGChainWrapper makes the graph behave like a normal Python callable:
    #   - Accepts a simple input dict with "question"
    #   - Builds the state required by the graph
    #   - Invokes the graph nodes internally (retrieve_docs -> generate_answer)
    #   - Returns a standard dict with 'answer' and 'source_documents'
    # This hides LangGraph complexity from app.py and allows seamless integration.
    class RAGChainWrapper:
        def __init__(self, graph):
            self.graph = graph

        def __call__(self, inputs: dict):
            question = inputs.get("question", "")
            state = {"question": question, "chat_history": []}
            result = self.graph.invoke(
                state,
                config={"configurable": {"thread_id": "default"}}
            )
            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("source_documents", [])
            }

    rag_chain = RAGChainWrapper(graph)

    return llm, retriever, rag_chain
