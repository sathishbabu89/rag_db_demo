
## **1. Suggested Folder Structure**

```
rag_app/
│
├── app.py                  # Main Streamlit RAG chatbot
├── requirements.txt        # Python dependencies
├── documents.db            # SQLite database (optional, auto-created)
├── .env                    # Your OpenRouter API key
├── README.md
└── .gitignore
```

---

## **2. README.md**

````markdown
# RAG Chatbot with SQLite + ChromaDB + OpenRouter Grok

📚 **Retrieval-Augmented Generation (RAG) Chatbot**  
This project demonstrates a RAG system where documents stored in a **SQLite database** are embedded using **Sentence Transformers**, stored in **ChromaDB** for fast similarity search, and queried using **OpenRouter Grok LLM** via a **Streamlit web interface**.

## Features

- Upload PDFs or paste manual text.
- Store documents in SQLite for persistence.
- Compute embeddings with Sentence Transformers (`all-MiniLM-L6-v2`).
- Store embeddings in ChromaDB for fast retrieval.
- Ask questions in a chat interface — Grok responds using relevant context.
- Session-based deduplication of documents.
- Lightweight, local development setup.

## Usage

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd rag_app
````

2. Create a `.env` file with your OpenRouter API key:

   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Upload PDF files or enter text manually, then ask questions in the chat.

## Requirements

* Python 3.10+
* Streamlit, ChromaDB, Sentence Transformers, PyPDF2, OpenAI SDK

````

---

## **3. requirements.txt**

```text
streamlit==1.28.0
openai==1.29.0
python-dotenv==1.0.0
chromadb==0.4.4
PyPDF2==3.0.2
sentence-transformers==2.2.2
````

*(Optional for OCR later: `pytesseract` and `Pillow`)*

---

## **4. .gitignore**

```
__pycache__/
*.pyc
.env
documents.db
```

* **`.env`** is ignored for security (API keys).
* **`documents.db`** is ignored because it’s generated automatically.

---

If you want, I can **also create a ready-to-use `git commit` setup** with **all these files pre-populated**, so you can directly push it to GitHub.

