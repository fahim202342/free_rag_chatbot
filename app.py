import os
import sys

# FIX: Protobuf version conflict — MUST be before any other imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tempfile
import hashlib
import json
import shutil
import time
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from operator import itemgetter
import pandas as pd


# ========== LOAD API KEYS ==========
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ========== PAGE SETUP ==========
st.set_page_config(page_title="Free RAG Chatbot", page_icon="🤖")

st.title("🤖 Free RAG Chatbot")
st.caption("100% Free | Powered by Groq + ChromaDB")

# ========== API CHECK ==========
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing!")
    st.info("Add it in Streamlit Cloud: Settings → Secrets")
    st.code('GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"', language="toml")
    st.stop()

# ========== PATHS ==========
CHROMA_DB_PATH = os.path.abspath("./chroma_db")
META_FILE = os.path.abspath("./uploaded_files_meta.json")


def load_uploaded_meta():
    """Load uploaded files metadata safely."""
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_uploaded_meta(meta):
    """Save uploaded files metadata safely."""
    try:
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save meta: {e}")


def safe_delete_chroma_db():
    """Safely delete Chroma DB."""
    if not os.path.exists(CHROMA_DB_PATH):
        return True

    # Try to delete collection first
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            client.delete_collection("docs")
        except Exception:
            pass
        del client
    except Exception:
        pass

    time.sleep(0.3)

    # Try deleting folder
    for attempt in range(5):
        try:
            shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
            if not os.path.exists(CHROMA_DB_PATH):
                return True
        except Exception:
            time.sleep(0.3)

    # Rename and delete as last resort
    try:
        temp_path = CHROMA_DB_PATH + "_old_" + str(int(time.time()))
        os.rename(CHROMA_DB_PATH, temp_path)
        shutil.rmtree(temp_path, ignore_errors=True)
        return True
    except Exception:
        return False


# ========== SIDEBAR ==========
uploaded_meta = load_uploaded_meta()

with st.sidebar:
    st.header("📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or JSON",
        type=["pdf", "json"],
        accept_multiple_files=True,
        key="doc_uploader"
    )

    # Process uploads
    if uploaded_files:
        process_btn = st.button("🚀 Process Files", type="primary", use_container_width=True)

        if process_btn:
            with st.spinner("Loading embedding model..."):
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True}
                    )
                except Exception as e:
                    st.error(f"❌ Embedding model failed: {e}")
                    st.stop()

            with st.spinner("Connecting to ChromaDB..."):
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                    vectordb = Chroma(
                        client=client,
                        embedding_function=embeddings,
                        collection_name="docs"
                    )
                except Exception as e:
                    st.error(f"❌ ChromaDB failed: {e}")
                    st.stop()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )

            new_files = []
            progress_bar = st.progress(0)

            for idx, file in enumerate(uploaded_files):
                progress_bar.progress((idx) / len(uploaded_files))

                file_hash = hashlib.md5(file.getvalue()).hexdigest()
                if file_hash in uploaded_meta:
                    continue

                file_ext = os.path.splitext(file.name)[1].lower()

                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                try:
                    if file_ext == ".pdf":
                        loader = PyMuPDFLoader(tmp_path)
                        docs = loader.load()
                    elif file_ext == ".json":
                        with open(tmp_path, "r", encoding="utf-8") as jf:
                            data = json.load(jf)
                        if isinstance(data, list):
                            text_content = "\n\n".join(
                                [json.dumps(item, ensure_ascii=False) for item in data]
                            )
                        else:
                            text_content = json.dumps(data, ensure_ascii=False)
                        docs = [Document(
                            page_content=text_content,
                            metadata={"source": file.name, "page": 1}
                        )]
                    else:
                        docs = []

                    if not docs:
                        st.warning(f"⚠️ Empty: {file.name}")
                        continue

                    chunks = text_splitter.split_documents(docs)
                    for chunk in chunks:
                        chunk.metadata["source"] = file.name

                    vectordb.add_documents(chunks)
                    uploaded_meta[file_hash] = file.name
                    new_files.append(file.name)

                except Exception as e:
                    st.error(f"❌ Failed {file.name}: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            progress_bar.empty()

            if new_files:
                save_uploaded_meta(uploaded_meta)
                st.success(f"✅ Done: {', '.join(new_files)}")
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
            else:
                st.info("ℹ️ All files already uploaded or empty.")

    # Show uploaded files
    if uploaded_meta:
        st.subheader("📁 Files")
        for fh, fname in list(uploaded_meta.items())[:20]:
            st.text(f"• {fname}")
        if len(uploaded_meta) > 20:
            st.text(f"... and {len(uploaded_meta) - 20} more")

    # Clear button
    if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
        with st.spinner("Clearing..."):
            success = safe_delete_chroma_db()
            if os.path.exists(META_FILE):
                try:
                    os.remove(META_FILE)
                except Exception:
                    pass
            st.cache_resource.clear()

        if success:
            st.success("Cleared!")
        else:
            st.warning("⚠️ Partially cleared. Restart app if needed.")
        time.sleep(1)
        st.rerun()

# ========== RETRIEVER ==========
MODEL = "llama-3.3-70b-versatile"
TEMP = 0.1
K_DOCS = 3


@st.cache_resource(ttl="1h")
def get_retriever(k):
    """Get retriever from existing ChromaDB."""
    if not os.path.exists(CHROMA_DB_PATH):
        return None
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        vectordb = Chroma(
            client=client,
            embedding_function=embeddings,
            collection_name="docs"
        )
        return vectordb.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        st.error(f"Retriever error: {e}")
        return None


retriever = get_retriever(K_DOCS)
has_docs = retriever is not None

if not has_docs:
    st.info("📤 Upload PDF/JSON from sidebar to start chatting.")

# Dummy retriever for when no docs exist
class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query):
        return []

if retriever is None:
    retriever = DummyRetriever()

# ========== LLM ==========
try:
    llm = ChatGroq(
        model_name=MODEL,
        temperature=TEMP,
        streaming=True,
        api_key=GROQ_API_KEY
    )
except Exception as e:
    st.error(f"❌ LLM failed: {e}")
    st.stop()

# ========== PROMPT & CHAIN ==========
qa_template = """You are a helpful AI assistant. Use ONLY the following context to answer the question. If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question: {question}

Answer:"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)


def format_docs(docs):
    if not docs:
        return "[No documents available. Please upload files from the sidebar.]"
    return "\n\n".join([d.page_content for d in docs])


qa_rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question")
    }
    | qa_prompt
    | llm
)

# ========== HANDLERS ==========
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
        self.sources = []
        for d in documents:
            meta = {
                "source": d.metadata.get("source", "Unknown"),
                "page": d.metadata.get("page", "N/A"),
                "content": d.page_content[:300]
            }
            self.sources.append(meta)

    def on_llm_end(self, response, **kwargs):
        if self.sources:
            with self.placeholder.container():
                st.markdown("---")
                st.markdown("**📚 Sources:**")
                st.dataframe(
                    pd.DataFrame(self.sources[:K_DOCS]),
                    use_container_width=True,
                    hide_index=True
                )


# ========== CHAT UI ==========
history = StreamlitChatMessageHistory(key="chat_history")

if len(history.messages) == 0:
    history.add_ai_message("Hello! 👋 Upload PDF or JSON files from the sidebar, then ask me anything about them.")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# Chat input — always visible
user_prompt = st.chat_input("Ask a question...", key="chat_input")

if user_prompt:
    st.chat_message("human").write(user_prompt)

    with st.chat_message("ai"):
        stream_container = st.empty()
        sources_placeholder = st.empty()

        stream_handler = StreamHandler(stream_container)
        pm_handler = PostMessageHandler(sources_placeholder)

        try:
            with st.spinner("Thinking..."):
                response = qa_rag_chain.invoke(
                    {"question": user_prompt},
                    config={"callbacks": [stream_handler, pm_handler]}
                )
        except Exception as e:
            st.error(f"❌ Error: {e}")
