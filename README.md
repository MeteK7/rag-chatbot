# 💬 MoodMate

**MoodMate** is an AI-powered assistant built on a **Retrieval Augmented Generation (RAG)** architecture and **Google Gemini**.  

It provides **reflective and supportive guidance** on mental wellness, personal growth, and self-improvement; all powered by real-world therapy datasets.

**⚠️ Note:** MoodMate is **not a licensed therapist.** It offers informational and reflective guidance only.

This application blends **retrieval-based reasoning** with **generative capabilities**, ensuring that responses are both **factual** and **insightful** — powered by the synergy of **LangChain**, **Hugging Face**, and **Gemini**.
---

## 🌟 Deploy Link

**DEPLOY LINK:** [github.com/MeteK7](https://github.com/MeteK7)
---

## 🌟 Key Highlights

- 🧠 **Dual Answering System:** Choose between dataset-based and reasoning-based answers  
- 📚 **Retrieval-Augmented Generation (RAG):** Combines retrieved data with Gemini reasoning  
- 🔍 **Chroma Vector Database:** Fast semantic search with Hugging Face embeddings  
- 🪄 **Streamlit UI:** Modern, interactive, and visually polished interface  
- 🔑 **.env Support:** Securely load your API keys for Hugging Face and Google Gemini  
- ⚙️ **Automatic vs Manual Mode:** Flexibility in how answers are generated  

---

## 🧩 How It Works

**MoodMate** implements a full **RAG pipeline** under the hood:

1. **Dataset Loading:** Fetches the `fadodr/mental_health_therapy` dataset from Hugging Face  
2. **Text Processing:** Cleans and splits data into retrievable text chunks  
3. **Vector Embedding:** Converts text into semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`  
4. **Vector Storage:** Stores embeddings locally using **Chroma DB**  
5. **Retriever:** Finds the most relevant context for your query  
6. **Gemini LLM (models/gemini-2.5-flash):** Generates high-quality responses using retrieved context  
7. **Streamlit Frontend:** Displays results beautifully with toggles, options, and expandable document previews  

---

## 🧠 Technologies Used

| Category | Tools & Frameworks |
|-----------|--------------------|
| **Frontend / UI** | Streamlit, streamlit-extras |
| **Backend / Logic** | Python, LangChain |
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | Hugging Face (`all-MiniLM-L6-v2`) |
| **Vector Database** | Chroma |
| **Dataset** | fadodr/mental_health_therapy |
| **Environment Management** | python-dotenv, venv |

---

## ⚙️ Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/MeteK7/rag-chatbot
cd moodmate
```

### 2. Virtual Environment Setup
```bash
# Create venv
python -m venv venv

# Activate
venv\Scripts\activate        # on Windows
source venv/bin/activate     # on Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
---

## 🔐 Setup API Keys

Create a `.env` file in the **root directory**:

```
GOOGLE_API_KEY=your_google_api_key
HF_TOKEN=your_huggingface_token
```

These keys enable Gemini and Hugging Face integrations.

---

## 🚀 Run the Application

```bash
streamlit run app.py
```

Then open your browser and go to:
```
http://localhost:8501
```

---

## 🗂️ Project Structure

```
personalized-lifecycle-companion/
├── app.py                   # Streamlit app (frontend + logic)
├── rag_pipeline.py           # RAG pipeline builder (retrieval + embeddings + LLM)
├── requirements.txt          
├── chroma_db/                # Vector database (auto-created)
├── .env                      # API keys (not tracked by Git)
└── README.md
```

---

## 💡 Use Cases

- 🧭 **Personal Development:** Ask for advice on habits, motivation, or communication  
- 💬 **Social Growth:** Learn empathy, relationships, or conflict resolution techniques  
- 💼 **Business Mindset:** Explore leadership, productivity, and growth strategies  
- 🧘 **Mental Wellness:** Engage in reflective, supportive conversations  

---

## 🧾 Example Queries

-“What are ways to practice self-care daily?”
- “What’s a good way to manage anxiety before public speaking?”  
- “How can I lead my team more effectively?”  

---

## 🧰 Future Enhancements

- Add multilingual support (Turkish, English, etc.)  
- Enable user-uploaded datasets for custom domains  
- Integrate long-term memory and adaptive suggestions  
- Deploy on Render / Hugging Face Spaces  

---

## 🤝 Contributing

Contributions are welcome!  
If you’d like to improve functionality, fix a bug, or enhance the UI:

1. Fork this repo  
2. Create your feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request  

---

## 📬 Contact

**Developer:** Mete Kaba  
**GitHub:** [github.com/MeteK7](https://github.com/MeteK7)  
**LinkedIn:** [/www.linkedin.com/in/mete-kaba-89b5221b8](https://www.linkedin.com/in/mete-kaba-89b5221b8)
