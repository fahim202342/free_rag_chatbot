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


# ========== SESSION STATE ==========
if "processed" not in st.session_state:
    st.session_state.processed = False
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "mobile_files" not in st.session_state:
    st.session_state.mobile_files = []

uploaded_meta = load_meta()


# ========== MOBILE-FRIENDLY FILE UPLOAD ==========
# Use HTML5 file input + base64 encoding for mobile compatibility

upload_html = """
<style>
.upload-area {
    border: 2px dashed #4CAF50;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    background: #f8fff8;
    margin: 10px 0;
}
.upload-area:active {
    background: #e8f5e9;
}
.file-input {
    display: none;
}
.upload-btn {
    background: #4CAF50;
    color: white;
    padding: 12px 24px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
    width: 100%;
}
.file-list {
    margin-top: 10px;
    font-size: 14px;
    color: #333;
}
</style>

<div class="upload-area">
    <input type="file" id="fileInput" class="file-input" accept=".pdf" multiple 
           onchange="handleFiles(this.files)">
    <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
        📎 Select PDF Files
    </button>
    <div id="fileList" class="file-list"></div>
</div>

<script>
let selectedFiles = [];
let fileData = {};

function handleFiles(files) {
    selectedFiles = Array.from(files);
    const list = document.getElementById('fileList');
    list.innerHTML = '<p><strong>Selected:</strong></p>';
    
    let processed = 0;
    
    selectedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            // Store base64 data with filename
            const base64 = e.target.result.split(',')[1];
            fileData[file.name] = base64;
            
            list.innerHTML += `<div>• ${file.name} (${(file.size/1024).toFixed(1)} KB)</div>`;
            
            processed++;
            if (processed === selectedFiles.length) {
                // Send to Streamlit
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: JSON.stringify(fileData)
                }, '*');
            }
        };
        reader.readAsDataURL(file);
    });
}
</script>
"""

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("📤 Upload Documents")
    st.caption(f"🔐 Session: `{USER_ID}`")
    st.divider()
    
    # MOBILE FIX: Use components.html for native file picker
    import streamlit.components.v1 as components
    
    # Container for file upload
    file_container = st.container()
    
    with file_container:
        # Try native HTML5 upload first (works better on mobile)
        st.markdown("### 📱 Mobile Upload")
        uploaded_data = components.html(upload_html, height=200, scrolling=False)
        
        # Fallback: Standard Streamlit uploader for desktop
        st.markdown("### 💻 Desktop Upload")
        standard_files = st.file_uploader(
            "Or use standard uploader",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"std_uploader_{USER_ID}_{st.session_state.upload_key}",
            label_visibility="collapsed"
        )
    
    # Collect files from both sources
    all_files = []
    
    # Process HTML5 uploaded files (mobile)
    if uploaded_data and isinstance(uploaded_data, str):
        try:
            import json as json_mod
            file_dict = json_mod.loads(uploaded_data)
            for fname, fbase64 in file_dict.items():
                file_bytes = base64.b64decode(fbase64)
                # Create temp file
                tmp_path = os.path.join(tempfile.gettempdir(), f"mobile_{USER_ID}_{fname}")
                with open(tmp_path, 'wb') as f:
                    f.write(file_bytes)
                all_files.append({
                    'name': fname,
                    'path': tmp_path,
                    'size': len(file_bytes),
                    'source': 'mobile'
                })
        except Exception as e:
            st.error(f"Mobile upload error: {e}")
    
    # Process standard uploaded files (desktop)
    if standard_files:
        for f in standard_files:
            tmp_path = os.path.join(tempfile.gettempdir(), f"desktop_{USER_ID}_{f.name}")
            with open(tmp_path, 'wb') as tmp:
                tmp.write(f.getvalue())
            all_files.append({
                'name': f.name,
                'path': tmp_path,
                'size': f.size,
                'source': 'desktop'
            })
    
    # Show selected files
    if all_files:
        st.session_state.mobile_files = all_files
        st.info(f"📎 {len(all_files)} file(s) ready")
        for f in all_files:
            st.caption(f"• {f['name']} ({f['size']/1024:.0f} KB)")
    
    # Process button
    process_btn = st.button(
        "🚀 Process Documents",
        type="primary",
        use_container_width=True,
        disabled=not all_files
    )
    
    # Show processed files
    if uploaded_meta:
        st.subheader("📁 Your Files")
        for fname in uploaded_meta.values():
            st.text(f"✅ {fname}")
    
    # Clear button
    if st.button("🗑️ Clear My Documents", use_container_width=True):
        safe_delete_user_db()
        st.session_state.processed = False
        st.session_state.upload_key += 1
        st.session_state.mobile_files = []
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    st.caption("🔒 Your files are private")


# ========== PROCESSING ==========
def process_files_with_gc(files_info, meta):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
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
    
    for i, finfo in enumerate(files_info):
        status.text(f"⏳ {i+1}/{len(files_info)}: {finfo['name']}")
        
        # Read file bytes
        with open(finfo['path'], 'rb') as f:
            file_bytes = f.read()
        
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if file_hash in meta:
            progress.progress((i + 1) / len(files_info))
            continue
        
        try:
            loader = PyMuPDFLoader(finfo['path'])
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            
            for chunk in chunks:
                chunk.metadata["source"] = finfo['name']
            
            vectordb.add_documents(chunks)
            meta[file_hash] = finfo['name']
            new_files.append(finfo['name'])
            
        except Exception as e:
            st.error(f"❌ {finfo['name']}: {str(e)}")
        finally:
            # Clean temp file
            try:
                os.unlink(finfo['path'])
            except:
                pass
            gc.collect()
        
        progress.progress((i + 1) / len(files_info))
    
    progress.empty()
    status.empty()
    return new_files, meta


if process_btn and st.session_state.mobile_files:
    st.subheader("🔄 Processing...")
    
    new_files, uploaded_meta = process_files_with_gc(
        st.session_state.mobile_files,
        uploaded_meta
    )
    
    if new_files:
        save_meta(uploaded_meta)
        st.success(f"✅ Done: {', '.join(new_files)}")
        st.session_state.processed = True
        st.session_state.mobile_files = []
        st.cache_resource.clear()
        st.balloons()
        st.rerun()
    else:
        st.info("ℹ️ Already processed")
        st.session_state.mobile_files = []


# ========== RETRIEVER ==========
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
    vectordb = Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name="docs"
    )
    return vectordb.as_retriever(search_kwargs={"k": K_DOCS})


retriever = get_retriever(USER_ID)

if retriever is None:
    st.info("📤 Upload PDFs and click 'Process Documents'")
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
