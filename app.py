import streamlit as st
from rag_pipeline import build_rag_pipeline
from streamlit_extras.add_vertical_space import add_vertical_space

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="💬 Personalized Lifecycle Companion",
    page_icon="💫",
    layout="centered",
)

# --- CUSTOM CSS (bubbles + badge) ---
st.markdown("""
<style>
    .main-title { text-align: center; font-size: 2.2em; font-weight: 700; color: #4A90E2; }
    .subtitle { text-align: center; font-size: 1.1em; color: #666; margin-bottom: 1.5em; }
    .user-bubble {
        background: linear-gradient(180deg, #dbe9ff, #c7ddff);
        padding: 0.75em 1em;
        border-radius: 12px;
        margin: 0.5em 0 0.25em 0;
        max-width: 80%;
    }
    .assistant-bubble {
        background: #f5f7fa;
        padding: 0.9em 1em;
        border-radius: 12px;
        margin: 0.25em 0 0.8em 0;
        border: 1px solid #e1e4e8;
        max-width: 80%;
    }
    .meta-badge {
        display: inline-block;
        font-size: 0.72em;
        padding: 2px 8px;
        border-radius: 999px;
        margin-left: 8px;
        vertical-align: middle;
    }
    .badge-dataset { background: #fff6ea; color: #b36b00; border: 1px solid #f0e68c; }
    .badge-general { background: #eefcf3; color: #0a7f53; border: 1px solid #bfead4; }
    .doc-box { background-color: #fffbe6; padding: 0.6em 0.8em; border-radius: 8px; border: 1px solid #f0e68c; margin-bottom: 0.5em; }
    .doc-q { font-weight: 600; color: #333; }
    .doc-a { color: #555; }

    /* Make chat area scrollable and avoid hiding under input */
    .chat-area {
        max-height: 70vh;
        overflow-y: auto;
        padding-right: 8px;
        padding-bottom: 120px; /* Space for input bar */
    }

    /* Fix the input container at the bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #ffffff;
        padding: 1rem 2rem;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
        z-index: 999;
    }

    /* Optional: make buttons line up neatly */
    .stButton button {
        height: 2.5em;
    }

    /* Hide Streamlit footer and hamburger for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-title">💬 MoodMate</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything about personal, social, or business growth — powered by RAG + Gemini</div>', unsafe_allow_html=True)

add_vertical_space(2)

# --- LOAD PIPELINE ---
@st.cache_resource
def load_chain():
    return build_rag_pipeline()

llm, retriever, rag_chain = load_chain()

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

# --- SESSION STATE MEMORY ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ensure input_box key exists so it persists across runs
if "input_box" not in st.session_state:
    st.session_state.input_box = ""

# --- LAYOUT: chat area + input at bottom ---
chat_col = st.container()

# Render chat area (so it updates live on each run)
with chat_col:
    st.markdown("## 💬 Conversation")
    chat_area = st.container()
    with chat_area:
        # Render each turn in order
        for i, turn in enumerate(st.session_state.chat_history):
            # User bubble (left)
            st.markdown(f'<div class="user-bubble">🧑 You: {turn["user"]}</div>', unsafe_allow_html=True)

            # Assistant bubble with subtle badge
            typ = turn.get("type", "General Reasoning")
            badge_html = (
                f'<span class="meta-badge badge-dataset">Dataset-Based</span>'
                if typ == "Dataset-Based Answer"
                else f'<span class="meta-badge badge-general">General Reasoning</span>'
            )

            st.markdown(f'<div class="assistant-bubble">🤖 Assistant: {turn["ai"]} {badge_html}</div>', unsafe_allow_html=True)

            # If dataset-based and has docs, show small expander for docs
            if turn.get("type") == "Dataset-Based Answer" and turn.get("docs"):
                with st.expander(f"📂 Top Retrieved Documents for message {i+1}"):
                    for d in turn["docs"][:3]:
                        parts = d.page_content.split("\n")
                        q_text = parts[0].replace("Q: ", "") if len(parts) > 0 else ""
                        a_text = parts[1].replace("A: ", "") if len(parts) > 1 else ""
                        st.markdown(
                            f'<div class="doc-box"><div class="doc-q">Q: {q_text}</div><div class="doc-a">A: {a_text}</div></div>',
                            unsafe_allow_html=True
                        )

# --- SEND CALLBACK LOGIC ---
def handle_send():
    query = st.session_state.input_box.strip()
    if not query:
        st.warning("Please enter a message.")
        return

    with st.spinner("🔍 Thinking and retrieving relevant information..."):
        # --- Build unified chat history for contextual prompting ---
        N_keep = 6  # Keep last 6 turns for context
        history_for_prompt = st.session_state.chat_history[-N_keep:]
        full_prompt = ""
        for turn in history_for_prompt:
            full_prompt += f"User: {turn['user']}\nAI: {turn['ai']}\n"
        full_prompt += f"User: {query}\nAI:"

        rag_answer, general_answer, docs = "", "", []

        # --- AUTO MODE ---
        if auto_mode:
            # Step 1: Try dataset-based (RAG) first
            rag_result = rag_chain({"question": query})
            rag_answer = rag_result.get("answer", "")
            docs = rag_result.get("source_documents", [])

            # Step 2: Evaluate RAG answer quality
            # Automatically decide whether to show the dataset-based answer or fall back to general reasoning
            # Explanation:
            # - any(kw in rag_answer.lower() for kw in fallback_keywords): checks if any "bad" keyword appears
            # - len(rag_answer.strip()) < 50: checks if the dataset-based answer is too short (likely low quality)
            # - not (...): inverts the condition — we show dataset answer only if it’s *good enough*            
            fallback_keywords = ["cannot answer", "no information", "based on the context", "i'm sorry"]
            rag_too_short = len(rag_answer.strip()) < 50
            rag_weak = any(kw in rag_answer.lower() for kw in fallback_keywords)

            if rag_weak or rag_too_short:
                # Step 3: Fallback to general reasoning ONLY if RAG is weak
                # Use full_prompt (last N_keep turns + current query) to generate answer with LLM
                general_response_obj = llm.invoke(full_prompt)
                general_answer = getattr(general_response_obj, "content", str(general_response_obj))
                chosen_answer = general_answer
                chosen_type = "General Reasoning"
            else:
                chosen_answer = rag_answer
                chosen_type = "Dataset-Based Answer"

        # --- MANUAL MODE ---
        else:
            if answer_type == "Dataset-Based Answer":
                rag_result = rag_chain({"question": query})
                rag_answer = rag_result.get("answer", "")
                docs = rag_result.get("source_documents", [])
                chosen_answer = rag_answer
                chosen_type = "Dataset-Based Answer"
            else:
                general_response_obj = llm.invoke(full_prompt)
                general_answer = getattr(general_response_obj, "content", str(general_response_obj))
                chosen_answer = general_answer
                chosen_type = "General Reasoning"

        # --- Append to unified chat history ---
        st.session_state.chat_history.append({
            "user": query,
            "ai": chosen_answer,
            "type": chosen_type,
            "docs": docs if chosen_type == "Dataset-Based Answer" else None
        })

    # ✅ Clear input after sending
    st.session_state.input_box = ""

# --- INPUT AREA (stays at bottom) ---
# --- FIXED INPUT BAR ---
st.markdown('<div class="input-container">', unsafe_allow_html=True)

query = st.text_input(
    "💭 Type your message here...",
    key="input_box",
    placeholder="e.g. How can I improve my communication skills?",
    label_visibility="collapsed"
)

col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.button("Send 💬", key="send_button", on_click=handle_send)
with col2:
    st.button("🧹 Clear Chat", key="clear_button", help="Clears conversation history (not persistent).", on_click=lambda: (
        st.session_state.chat_history.clear(),
        st.session_state.update({"input_box": ""}),
        st.rerun()
    ))

st.markdown('</div>', unsafe_allow_html=True)