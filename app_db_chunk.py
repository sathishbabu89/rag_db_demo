import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import sqlite3
from PyPDF2 import PdfReader

# -------------------------------
# 1. Load environment variables
# -------------------------------
load_dotenv()

# OpenRouter client for chat (Grok)
chat_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# -------------------------------
# 2. Setup SQLite DB (local dev)
# -------------------------------
conn = sqlite3.connect("documents.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    category TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# -------------------------------
# 3. Setup ChromaDB with Sentence Transformers
# -------------------------------
chroma_client = chromadb.Client()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="docs",
    embedding_function=embedding_fn
)

# -------------------------------
# 4. Chunking function
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -------------------------------
# 5. Streamlit UI
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot with SQLite + ChromaDB + Chunking")

st.sidebar.header("Add Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", accept_multiple_files=True, type=["pdf"]
)

manual_text = st.sidebar.text_area("Or paste text manually")

if "added_doc_ids" not in st.session_state:
    st.session_state.added_doc_ids = set()

# -------------------------------
# 6. Process uploaded PDFs
# -------------------------------
if uploaded_files:
    progress_bar = st.sidebar.progress(0)
    for i, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            pdf_reader = PdfReader(uploaded_file)
            full_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

            if full_text.strip():
                chunks = chunk_text(full_text)

                for chunk in chunks:
                    cursor.execute("INSERT INTO documents (title, content) VALUES (?, ?)",
                                   (uploaded_file.name, chunk))
                    conn.commit()
                    doc_id = cursor.lastrowid

                    collection.add(documents=[chunk], ids=[f"doc_{doc_id}"])
                    st.session_state.added_doc_ids.add(doc_id)

        except Exception as e:
            st.sidebar.warning(f"Failed to process {uploaded_file.name}: {e}")
        progress_bar.progress(i / len(uploaded_files))
    st.sidebar.success("PDF documents chunked and added ✅")

# -------------------------------
# 7. Process manual text input
# -------------------------------
if manual_text.strip():
    chunks = chunk_text(manual_text.strip())

    for chunk in chunks:
        cursor.execute("INSERT INTO documents (title, content) VALUES (?, ?)",
                       ("Manual Entry", chunk))
        conn.commit()
        doc_id = cursor.lastrowid

        collection.add(documents=[chunk], ids=[f"doc_{doc_id}"])
        st.session_state.added_doc_ids.add(doc_id)

    st.sidebar.success("Manual text chunked and added ✅")

# -------------------------------
# 8. Chat input and response
# -------------------------------
user_query = st.chat_input("Ask me something about your documents...")

if user_query:
    try:
        results = collection.query(query_texts=[user_query], n_results=3)
        retrieved_docs = results.get("documents", [[]])[0]
        context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
    except Exception as e:
        context = f"Error retrieving documents: {e}"

    prompt = f"Answer the question based on context below:\n\nContext: {context}\n\nQuestion: {user_query}"

    try:
        response = chat_client.chat.completions.create(
            model="x-ai/grok-4-fast:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers using the given context."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Sorry, I couldn't get a response from the model: {e}"

    with st.chat_message("user"):
        st.write(user_query)
    with st.chat_message("assistant"):
        st.write(answer)
