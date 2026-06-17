import streamlit as st
import os
import tempfile
import hashlib
import json
import shutil
import time
import gc

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from operator import itemgetter
import pandas as pd
import chromadb


# ========== LOAD API KEYS ==========
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ========== PAGE SETUP ==========
st.set_page_config(page_title="Free RAG Chatbot", page_icon="🤖")

st.title("🤖 Free RAG Chatbot")
st.caption("100% Free")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing! Add to `.env` or Streamlit Secrets")
    st.stop()

# ========== PATHS (Streamlit Cloud safe) ==========
CHROMA_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")
META_FILE = os.path.join(tempfile.gettempdir(), "uploaded_files_meta.json")


def load_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}


def save_meta(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f)


def safe_delete_db():
    if not os.path.exists(CHROMA_DB_PATH):
        return True
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            client.delete_collection("docs")
        except Exception:
            pass
        del client
    except Exception:
        pass
    time.sleep(0.5)
    for _ in range(5):
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            return True
        except PermissionError:
            time.sleep(0.5)
    return False


# ========== SESSION STATE ==========
if "processed" not in st.session_state:
    st.session_state.processed = False
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "pending_files" not in st.session_state:
    st.session_state.pending_files = None

uploaded_meta = load_meta()

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("📤 Upload Documents")
    
    # Step 1: Only select files (NO processing here!)
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_key}"
    )
    
    # Store files in session state for later processing
    if uploaded_files:
        st.session_state.pending_files = uploaded_files
        st.info(f"📎 {len(uploaded_files)} file(s) selected")
        for f in uploaded_files:
            st.caption(f"• {f.name} ({f.size/1024:.0f} KB)")
    
    # Step 2: PROCESS BUTTON - Critical for mobile!
    # Processing only happens when user clicks this
    process_btn = st.button(
        "🚀 Process Documents", 
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files
    )
    
    # Show already processed files
    if uploaded_meta:
        st.subheader("📁 Processed Files")
        for fname in uploaded_meta.values():
            st.text(f"✅ {fname}")
    
    # Clear button
    if st.button("🗑️ Clear All", use_container_width=True):
        safe_delete_db()
        if os.path.exists(META_FILE):
            os.remove(META_FILE)
        st.session_state.processed = False
        st.session_state.upload_key += 1
        st.session_state.pending_files = None
        st.cache_resource.clear()
        st.rerun()


# ========== PROCESSING FUNCTION (with memory cleanup) ==========
def process_files_with_gc(files, meta):
    """Process files one by one with garbage collection after each"""
    
    # Initialize embeddings (once)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        vectordb = Chroma(
            client=client,
            embedding_function=embeddings,
            collection_name="docs"
        )
    except Exception:
        vectordb = Chroma(
            client=client,
            embedding_function=embeddings,
            collection_name="docs",
            persist_directory=CHROMA_DB_PATH
        )
    
    progress = st.progress(0)
    status = st.empty()
    new_files = []
    
    for i, file in enumerate(files):
        # Show progress
        status.text(f"⏳ {i+1}/{len(files)}: {file.name}")
        
        # Skip if already processed
        file_hash = hashlib.md5(file.getvalue()).hexdigest()
        if file_hash in meta:
            progress.progress((i + 1) / len(files))
            continue
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Load PDF
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            
            # Split into chunks
            chunks = text_splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source"] = file.name
            
            # Add to vector DB
            vectordb.add_documents(chunks)
            
            # Mark as processed
            meta[file_hash] = file.name
            new_files.append(file.name)
            
        except Exception as e:
            st.error(f"❌ {file.name}: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            # CRITICAL: Free memory after EACH file (prevents mobile crash)
            gc.collect()
        
        progress.progress((i + 1) / len(files))
    
    progress.empty()
    status.empty()
    return new_files, meta


# ========== HANDLE PROCESSING (only when button clicked) ==========
if process_btn and st.session_state.pending_files:
    st.subheader("🔄 Processing...")
    
    new_files, uploaded_meta = process_files_with_gc(
        st.session_state.pending_files,
        uploaded_meta
    )
    
    if new_files:
        save_meta(uploaded_meta)
        st.success(f"✅ Processed: {', '.join(new_files)}")
        st.session_state.processed = True
        st.session_state.pending_files = None
        st.cache_resource.clear()
        st.balloons()
        st.rerun()
    else:
        st.info("ℹ️ All files already processed")
        st.session_state.pending_files = None


# ========== RETRIEVER ==========
MODEL = "llama-3.3-70b-versatile"
TEMP = 0.1
K_DOCS = 3


@st.cache_resource(ttl="1h")
def get_retriever():
    if not os.path.exists(CHROMA_DB_PATH):
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    vectordb = Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name="docs"
    )
    return vectordb.as_retriever(search_kwargs={"k": K_DOCS})


retriever = get_retriever()

if retriever is None:
    st.info("📤 Upload PDFs from sidebar and click 'Process Documents'")
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
    {"context": itemgetter("question") | retriever | format_docs,
     "question": itemgetter("question")}
    | qa_prompt
    | llm
)


# ========== STREAM HANDLERS ==========
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
            self.sources.append({
                "source": d.metadata.get("source", "Unknown"),
                "page": d.metadata.get("page", "N/A"),
                "content": d.page_content[:200]
            })

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
            qa_rag_chain.invoke(
                {"question": user_prompt},
                {"callbacks": [stream_handler, pm_handler]}
            )
        except Exception as e:
            st.error(f"Error: {e}")
