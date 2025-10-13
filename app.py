import streamlit as st
from rag_pipeline import build_rag_pipeline
from langchain.chains import RetrievalQA

st.set_page_config(page_title="💬 Personalized Lifecycle Companion")
st.title("💬 Personalized Lifecycle Companion")
st.write("Ask me anything about personal, social, or business growth.")

@st.cache_resource
def load_chain():
    # Returns llm, retriever, and qa_chain
    return build_rag_pipeline()

# Get LLM, retriever, and RAG chain
llm, retriever, qa_chain = load_chain()

query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking..."):
        # --- Step 1: Retrieve top documents ---
        docs = retriever.get_relevant_documents(query)

        # Step 2: Run RAG chain
        rag_answer = qa_chain.run(query)

        # Step 3: Determine if RAG answer is meaningful
        fallback_keywords = [
            "cannot answer",
            "no information",
            "based on the context",
            "i'm sorry"
        ]

        if any(kw in rag_answer.lower() for kw in fallback_keywords) or len(rag_answer.strip()) < 50:
            # --- Step 4: Fallback to general LLM reasoning ---
            response_obj = llm.invoke(query)
            general_answer = getattr(response_obj, "content", str(response_obj))
            st.success(f"🤖 **General reasoning answer:**\n\n{general_answer}")
        else:
            st.success(f"📚 **Dataset-based answer:**\n\n{rag_answer}")

        # Optional: show top retrieved snippets
        st.write("**Top retrieved documents:**")
        for i, d in enumerate(docs[:3]):
            st.markdown(f"**Doc {i+1}:** {d.page_content[:300]}...")
