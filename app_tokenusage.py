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

chat_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# -------------------------------
# 2. Setup SQLite DB
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
# 3. Setup ChromaDB
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
# 4. Streamlit UI
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot with Cost Tracking")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", accept_multiple_files=True, type=["pdf"]
)
manual_text = st.sidebar.text_area("Or paste text manually")

if "added_doc_ids" not in st.session_state:
    st.session_state.added_doc_ids = set()

# -------------------------------
# 5. Process PDFs
# -------------------------------
if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            if text.strip():
                cursor.execute("INSERT INTO documents (title, content) VALUES (?, ?)",
                               (uploaded_file.name, text))
                conn.commit()
                doc_id = cursor.lastrowid
                collection.add(documents=[text], ids=[f"doc_{doc_id}"])
                st.session_state.added_doc_ids.add(doc_id)
        except Exception as e:
            st.sidebar.warning(f"Failed to process {uploaded_file.name}: {e}")
    st.sidebar.success("PDF documents added ✅")

# -------------------------------
# 6. Manual text input
# -------------------------------
if manual_text.strip():
    cursor.execute("INSERT INTO documents (title, content) VALUES (?, ?)",
                   ("Manual Entry", manual_text.strip()))
    conn.commit()
    doc_id = cursor.lastrowid
    collection.add(documents=[manual_text.strip()], ids=[f"doc_{doc_id}"])
    st.session_state.added_doc_ids.add(doc_id)
    st.sidebar.success("Manual text added ✅")

# -------------------------------
# 7. Chat input
# -------------------------------
user_query = st.chat_input("Ask me something about your documents...")

if user_query:
    try:
        results = collection.query(query_texts=[user_query], n_results=2)
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

        # ✅ Extract token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # ⚡ Approximate cost calculation (example: $0.000002 per token)
        # Replace with actual model pricing from OpenRouter
        cost_per_token = 0.000002  
        total_cost = total_tokens * cost_per_token

    except Exception as e:
        answer = f"Sorry, I couldn't get a response: {e}"
        prompt_tokens = completion_tokens = total_tokens = total_cost = 0

    # Display chat
    with st.chat_message("user"):
        st.write(user_query)
    with st.chat_message("assistant"):
        st.write(answer)

    # Show cost details
    with st.expander("📊 Token Usage & Cost Details"):
        st.write(f"**Prompt tokens:** {prompt_tokens}")
        st.write(f"**Completion tokens:** {completion_tokens}")
        st.write(f"**Total tokens:** {total_tokens}")
        st.write(f"**Estimated cost:** ${total_cost:.6f}")
