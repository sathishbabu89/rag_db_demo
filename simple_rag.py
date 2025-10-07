import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions
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
# 2. Setup ChromaDB with Sentence Transformers
# -------------------------------
chroma_client = chromadb.Client()

# Use model name string, ChromaDB will load internally
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="docs",
    embedding_function=embedding_fn
)

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 Retrieval-Augmented Generation (RAG) Chatbot")

st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF files", accept_multiple_files=True, type=["pdf"]
)

# -------------------------------
# 4. Track added document IDs (session-based)
# -------------------------------
if "added_doc_ids" not in st.session_state:
    st.session_state.added_doc_ids = set()

# -------------------------------
# 5. Extract and store text from PDFs
# -------------------------------
if uploaded_files:
    progress_bar = st.sidebar.progress(0)
    for i, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

            if text.strip():
                doc_id = f"doc_{uploaded_file.name}_{i}"

                if doc_id not in st.session_state.added_doc_ids:
                    collection.add(documents=[text], ids=[doc_id])
                    st.session_state.added_doc_ids.add(doc_id)
        except Exception as e:
            st.sidebar.warning(f"Failed to process {uploaded_file.name}: {e}")

        progress_bar.progress(i / len(uploaded_files))
    st.sidebar.success("Documents added to knowledge base ✅")

# -------------------------------
# 6. Chat input and response
# -------------------------------
user_query = st.chat_input("Ask me something about your documents...")

if user_query:
    # Retrieve relevant docs safely
    try:
        results = collection.query(query_texts=[user_query], n_results=2)
        retrieved_docs = results.get("documents", [[]])[0]
        context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
    except Exception as e:
        context = f"Error retrieving documents: {e}"

    # Build prompt
    prompt = f"Answer the question based on context below:\n\nContext: {context}\n\nQuestion: {user_query}"

    # Query Grok LLM safely
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

    # Show results in chat
    with st.chat_message("user"):
        st.write(user_query)
    with st.chat_message("assistant"):
        st.write(answer)
