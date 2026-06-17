import streamlit as st
import os
import tempfile
import hashlib
import json
import shutil
import time
import gc
import uuid
import base64
import io

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

st.set_page_config(page_title="Free RAG Chatbot", page_icon="🤖")

st.title("🤖 Free RAG Chatbot")
st.caption("100% Free")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing!")
    st.stop()

# ========== USER ISOLATION ==========
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:12]

USER_ID = st.session_state.user_id
BASE_DIR = os.path.join(tempfile.gettempdir(), "rag_users")
USER_DIR = os.path.join(BASE_DIR, USER_ID)
CHROMA_DB_PATH = os.path.join(USER_DIR, "chroma_db")
META_FILE = os.path.join(USER_DIR, "meta.json")
os.makedirs(USER_DIR, exist_ok=True)


def load_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}


def save_meta(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f)


def safe_delete_user_db():
    if not os.path.exists(USER_DIR):
        return True
    try:
        if os.path.exists(CHROMA_DB_PATH):
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            try:
                client.delete_collection("docs")
            except Exception:
                pass
            del client
            time.sleep(0.3)
    except Exception:
        pass
    for _ in range(5):
        try:
            shutil.rmtree(USER_DIR)
            return True
        except PermissionError:
            time.sleep(0.5)
    return False


uploaded_meta = load_meta()

# ========== CRITICAL: MOBILE FILE UPLOAD ==========
# Use query params to receive files from JavaScript
# This avoids st.file_uploader completely on mobile

query_params = st.query_params
file_data_json = query_params.get("files", None)

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("📤 Upload Documents")
    st.caption(f"🔐 Session: `{USER_ID}`")
    st.divider()
    
    # Method 1: HTML5 File Input (Mobile Compatible)
    st.markdown("### 📱 Upload PDF")
    
    # Create a form that submits via URL params (no page reload)
    upload_component = """
    <form id="uploadForm" style="margin:0;padding:0;">
        <input type="file" id="pdfInput" accept="application/pdf" style="display:none;" 
               onchange="uploadFiles()">
        <button type="button" onclick="document.getElementById('pdfInput').click()" 
                style="width:100%;padding:12px;background:#4CAF50;color:white;
                       border:none;border-radius:8px;font-size:16px;cursor:pointer;">
            📎 Select PDF File
        </button>
        <div id="status" style="margin-top:10px;font-size:14px;color:#666;"></div>
    </form>
    
    <script>
    async function uploadFiles() {
        const input = document.getElementById('pdfInput');
        const file = input.files[0];
        if (!file) return;
        
        const status = document.getElementById('status');
        status.innerHTML = '⏳ Reading...';
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const base64 = e.target.result.split(',')[1];
            // Store in localStorage temporarily
            localStorage.setItem('pending_pdf', JSON.stringify({
                name: file.name,
                data: base64,
                size: file.size,
                user: '""" + USER_ID + """'
            }));
            status.innerHTML = '✅ Ready! Click Process below.';
            // Notify Streamlit
            window.parent.postMessage({type: 'streamlit:fileReady'}, '*');
        };
        reader.readAsDataURL(file);
    }
    </script>
    """
    
    st.components.v1.html(upload_component, height=120)
    
    # Check localStorage via another component
    check_component = """
    <script>
    (function() {
        const data = localStorage.getItem('pending_pdf');
        if (data) {
            const fileInfo = JSON.parse(data);
            // Send to Streamlit via URL param
            const currentUrl = new URL(window.location.href);
            currentUrl.searchParams.set('files', btoa(data));
            window.history.replaceState({}, '', currentUrl);
            localStorage.removeItem('pending_pdf');
        }
    })();
    </script>
    """
    st.components.v1.html(check_component, height=0)
    
    # Read file from query params
    pending_file = None
    if file_data_json:
        try:
            file_info = json.loads(base64.b64decode(file_data_json).decode())
            if file_info.get("user") == USER_ID:
                pending_file = file_info
                # Clear param
                st.query_params.clear()
        except Exception:
            pass
    
    # Show file info
    if pending_file:
        st.success(f"📎 {pending_file['name']}")
        st.info(f"Size: {pending_file['size']/1024:.1f} KB")
    
    # Process button
    process_btn = st.button(
        "🚀 Process Document",
        type="primary",
        use_container_width=True,
        disabled=not pending_file
    )
    
    # Show processed files
    if uploaded_meta:
        st.subheader("📁 Your Files")
        for fname in uploaded_meta.values():
            st.text(f"✅ {fname}")
    
    # Clear
    if st.button("🗑️ Clear My Documents", use_container_width=True):
        safe_delete_user_db()
        st.session_state.upload_key = 0
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    st.caption("🔒 Private & Isolated")

# ========== PROCESSING ==========
if process_btn and pending_file:
    st.subheader("🔄 Processing...")
    
    # Decode and save
    file_bytes = base64.b64decode(pending_file['data'])
    tmp_path = os.path.join(tempfile.gettempdir(), f"{USER_ID}_{pending_file['name']}")
    
    with open(tmp_path, 'wb') as f:
        f.write(file_bytes)
    
    # Process
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        vectordb = Chroma(client=client, embedding_function=embeddings, collection_name="docs")
    except Exception:
        vectordb = Chroma(client=client, embedding_function=embeddings, collection_name="docs", persist_directory=CHROMA_DB_PATH)
    
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    if file_hash not in uploaded_meta:
        with st.spinner("Loading PDF..."):
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            
            for chunk in chunks:
                chunk.metadata["source"] = pending_file['name']
            
            vectordb.add_documents(chunks)
            uploaded_meta[file_hash] = pending_file['name']
            save_meta(uploaded_meta)
            
            st.success(f"✅ {pending_file['name']} processed!")
            st.balloons()
    else:
        st.info("ℹ️ Already processed")
    
    # Cleanup
    try:
        os.unlink(tmp_path)
    except:
        pass
    gc.collect()
    
    # Clear pending
    pending_file = None
    st.cache_resource.clear()
    time.sleep(1)
    st.rerun()

# ========== RETRIEVER & CHAT ==========
MODEL = "llama-3.3-70b-versatile"
TEMP = 0.1
K_DOCS = 3


@st.cache_resource(ttl="1h")
def get_retriever(user_id):
    user_chroma = os.path.join(BASE_DIR, user_id, "chroma_db")
    if not os.path.exists(user_chroma):
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    client = chromadb.PersistentClient(path=user_chroma)
    vectordb = Chroma(client=client, embedding_function=embeddings, collection_name="docs")
    return vectordb.as_retriever(search_kwargs={"k": K_DOCS})


retriever = get_retriever(USER_ID)

if retriever is None:
    st.info("📤 Upload a PDF and click 'Process Document'")
    st.stop()

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


history = StreamlitChatMessageHistory(key=f"chat_{USER_ID}")

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
