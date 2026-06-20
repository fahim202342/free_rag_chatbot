import streamlit as st
import os
import tempfile
import hashlib
import json
import shutil
import time
import warnings

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from operator import itemgetter
import pandas as pd


# ========== LOAD API KEYS FROM .ENV ==========
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ========== PAGE SETUP ==========
st.set_page_config(page_title="Free RAG Chatbot", page_icon="🤖")

st.title("🤖 Free RAG Chatbot")
st.caption("100% Free")

# ========== API CHECK ==========
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing! Create a `.env` file with GROQ_API_KEY=your_key")
    st.stop()

# ========== PERSISTENT STORAGE SETUP ==========
CHROMA_DB_PATH = "./chroma_db"
META_FILE = "./uploaded_files_meta.json"
COLLECTION_NAME = "rag_docs_v2"


def load_uploaded_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}


def save_uploaded_meta(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f)


def safe_delete_chroma_db():
    """Safely delete Chroma DB."""
    if not os.path.exists(CHROMA_DB_PATH):
        return True

    time.sleep(0.5)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                try:
                    temp_path = CHROMA_DB_PATH + "_old"
                    if os.path.exists(temp_path):
                        shutil.rmtree(temp_path)
                    os.rename(CHROMA_DB_PATH, temp_path)
                    shutil.rmtree(temp_path, ignore_errors=True)
                    return True
                except Exception:
                    return False
    return False


# ========== AUTO-CLEAN CORRUPTED DB ON STARTUP ==========
def reset_db_state():
    """Delete DB and meta to force a clean start."""
    safe_delete_chroma_db()
    if os.path.exists(META_FILE):
        try:
            os.remove(META_FILE)
        except Exception:
            pass


# Always start fresh — no corrupted DB issues ever
reset_db_state()

uploaded_meta = load_uploaded_meta()

# ========== DOCUMENT UPLOAD SIDEBAR ==========
with st.sidebar:
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        st.subheader("Processing...")
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        vectordb = Chroma.from_documents(
            documents=[],
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        new_files = []
        for file in uploaded_files:
            file_hash = hashlib.md5(file.getvalue()).hexdigest()
            if file_hash in uploaded_meta:
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            try:
                loader = PyMuPDFLoader(tmp_path)
                docs = loader.load()
                chunks = text_splitter.split_documents(docs)

                for chunk in chunks:
                    chunk.metadata["source"] = file.name

                vectordb.add_documents(chunks)
                uploaded_meta[file_hash] = file.name
                new_files.append(file.name)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
            finally:
                os.unlink(tmp_path)

        if new_files:
            save_uploaded_meta(uploaded_meta)
            st.success(f"✅ Uploaded: {', '.join(new_files)}")
            st.rerun()
        else:
            st.info("ℹ️ All files already uploaded.")

    if uploaded_meta:
        st.subheader("📁 Uploaded Files")
        for fh, fname in uploaded_meta.items():
            st.text(f"• {fname}")

    if st.button("🗑️ Clear All Documents", type="secondary"):
        reset_db_state()
        st.success("Database cleared!")
        st.rerun()

# ========== RETRIEVER ==========
MODEL = "llama-3.3-70b-versatile"
TEMP = 0.1
K_DOCS = 3


@st.cache_resource(ttl="1h")
def configure_retriever(k, db_exists):
    if not db_exists:
        return None
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vectordb.as_retriever(search_kwargs={"k": k})


db_exists = os.path.exists(CHROMA_DB_PATH)
retriever = configure_retriever(K_DOCS, db_exists)

if retriever is None:
    st.info("📤 No documents uploaded yet. Please upload PDF files from the sidebar.")
    st.stop()

# ========== LLM & CHAIN ==========
llm = ChatGroq(model_name=MODEL, temperature=TEMP, streaming=True)

qa_template = """You are a helpful AI assistant. Use ONLY the following context to answer the question. If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question: {question}

Answer:"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


qa_rag_chain = (
    {"context": itemgetter("question") | retriever | format_docs, "question": itemgetter("question")}
    | qa_prompt
    | llm
)

# ========== STREAM HANDLER ==========


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)


class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.sources = []

    def on_retriever_end(self, documents, **kwargs):
        for d in documents:
            meta = {
                "source": d.metadata.get("source", "Unknown"),
                "page": d.metadata.get("page", "N/A"),
                "content": d.page_content[:200]
            }
            self.sources.append(meta)

    def on_llm_end(self, response, **kwargs):
        if self.sources:
            with self.placeholder.container():
                st.markdown("---")
                st.markdown("**Sources:**")
                st.dataframe(pd.DataFrame(self.sources[:3]), width=1000)


# ========== CHAT ==========
history = StreamlitChatMessageHistory(key="langchain_messages")

if len(history.messages) == 0:
    history.add_ai_message("Hello! Ask me anything about your documents.")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if user_prompt := st.chat_input("Ask a question..."):
    st.chat_message("human").write(user_prompt)

    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        sources_placeholder = st.empty()
        pm_handler = PostMessageHandler(sources_placeholder)

        try:
            response = qa_rag_chain.invoke(
                {"question": user_prompt},
                {"callbacks": [stream_handler, pm_handler]}
            )
        except Exception as e:
            st.error(f"Error: {e}")
