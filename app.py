import streamlit as st
from rag_pipeline import build_rag_pipeline
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="💬 Personalized Lifecycle Companion",
    page_icon="💫",
    layout="centered",
)

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.2em;
            font-weight: 700;
            color: #4A90E2;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1em;
            color: #666;
            margin-bottom: 1.5em;
        }
        .response-box {
            background-color: #f5f7fa;
            padding: 1em 1.2em;
            border-radius: 10px;
            border: 1px solid #e1e4e8;
            margin-bottom: 1em;
        }
        .doc-box {
            background-color: #fffbe6;
            padding: 0.8em 1em;
            border-radius: 8px;
            border: 1px solid #f0e68c;
            margin-bottom: 0.5em;
        }
        .doc-q { font-weight: 600; color: #333; }
        .doc-a { color: #555; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER ---
st.markdown('<div class="main-title">💬 Personalized Lifecycle Companion</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything about personal, social, or business growth — powered by RAG + Gemini</div>', unsafe_allow_html=True)

add_vertical_space(2)

# --- LOAD PIPELINE ---
@st.cache_resource
def load_chain():
    return build_rag_pipeline()

llm, retriever, qa_chain = load_chain()

# --- USER SETTINGS ---
st.markdown("### ⚙️ Answer Selection Settings")

# Automatic vs Manual mode
auto_mode = st.checkbox("Automatic answer selection (default)", value=True)

# Manual answer type selection appears only if auto_mode is off
if not auto_mode:
    answer_type = st.radio(
        "Select answer type:",
        ("Dataset-Based Answer", "General Reasoning Answer"),
        index=0
    )

add_vertical_space(1)

# --- INPUT AREA ---
query = st.text_input("💭 What would you like to ask?", placeholder="e.g. How can I improve my communication skills?")

add_vertical_space(1)

# --- SUBMIT BUTTON ---
if st.button("Ask 💬"):
    if query.strip() == "":
        st.warning("Please enter a question first!")
    else:
        with st.spinner("🔍 Thinking and retrieving relevant information..."):
            # Generate general answer in parallel (for quick switching)
            response_obj = llm.invoke(query)
            general_answer = getattr(response_obj, "content", str(response_obj))

            # --- Determine which answer to show ---
            show_dataset = True
            docs = []
            rag_answer = ""

            if auto_mode:
                # Retrieve dataset-based answer
                docs = retriever.get_relevant_documents(query)
                rag_answer = qa_chain.run(query)

                fallback_keywords = ["cannot answer", "no information", "based on the context", "i'm sorry"]
                show_dataset = not (any(kw in rag_answer.lower() for kw in fallback_keywords) or len(rag_answer.strip()) < 50)
            else:
                # Manual mode
                show_dataset = answer_type == "Dataset-Based Answer"
                if show_dataset:
                    docs = retriever.get_relevant_documents(query)
                    rag_answer = qa_chain.run(query)

            # --- Display selected answer ---
            if show_dataset:
                st.markdown("### 📚 Dataset-Based Answer")
                st.markdown(f'<div class="response-box">{rag_answer}</div>', unsafe_allow_html=True)
            else:
                st.markdown("### 🤖 General Reasoning Answer")
                st.markdown(f'<div class="response-box">{general_answer}</div>', unsafe_allow_html=True)

            # --- Optional: Show top retrieved documents ---
            if show_dataset and docs:
                with st.expander("📂 Show Top Retrieved Documents (for debugging)"):
                    for i, d in enumerate(docs[:3]):
                        parts = d.page_content.split("\n")
                        q_text = parts[0].replace("Q: ", "") if len(parts) > 0 else ""
                        a_text = parts[1].replace("A: ", "") if len(parts) > 1 else ""
                        st.markdown(
                            f'<div class="doc-box"><div class="doc-q">Q: {q_text}</div><div class="doc-a">A: {a_text}</div></div>',
                            unsafe_allow_html=True
                        )
