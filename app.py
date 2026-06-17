import streamlit as st
import os
import tempfile
import gc
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Free RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ============ CSS ============
st.markdown("""
<style>
    .stFileUploader {max-width: 100%;}
    .chat-message {padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem;}
    .user-message {background-color: #e3f2fd;}
    .bot-message {background-color: #f3e5f5;}
</style>
""", unsafe_allow_html=True)

# ============ SESSION STATE INIT ============
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

# ============ SIDEBAR ============
with st.sidebar:
    st.header("📄 Upload Documents")
    
    # File uploader with unique key to force refresh
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_key}"
    )
    
    st.divider()
    st.subheader("Processing Status")
    
    # Process button - IMPORTANT: Mobile friendly
    process_clicked = st.button(
        "🚀 Process Documents", 
        type="primary",
        use_container_width=True
    )
    
    st.divider()
    st.subheader("📁 Uploaded Files")
    
    if st.session_state.processed_files:
        for fname in st.session_state.processed_files:
            st.write(f"✅ {fname}")
    else:
        st.info("No files processed yet")
    
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.processed_files = set()
        st.session_state.conversation = None
        st.session_state.upload_key += 1  # Force uploader reset
        # Clean chroma temp dir
        chroma_path = os.path.join(tempfile.gettempdir(), "chroma_db")
        if os.path.exists(chroma_path):
            import shutil
            shutil.rmtree(chroma_path, ignore_errors=True)
        st.rerun()

# ============ MAIN CONTENT ============
st.title("🤖 Free RAG Chatbot")
st.caption("100% Free - Powered by Groq & HuggingFace")

# ============ PROCESSING LOGIC ============
def extract_text_from_pdf(file_bytes):
    """Memory-safe PDF text extraction"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        text = ""
        for page in doc:
            text += page.get_text()
            # Periodic GC to prevent mobile memory crash
            if len(text) > 500000:  # Every ~500KB
                gc.collect()
        return text
    finally:
        doc.close()
        gc.collect()

def process_documents(uploaded_files):
    """Process uploaded PDFs and create vector store"""
    if not uploaded_files:
        return False
    
    all_texts = []
    new_files = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    
    for idx, file in enumerate(uploaded_files):
        if file.name in st.session_state.processed_files:
            continue  # Skip already processed
        
        status_text.text(f"Processing: {file.name} ({idx+1}/{total_files})")
        
        try:
            file_bytes = file.getvalue()
            text = extract_text_from_pdf(file_bytes)
            
            if text.strip():
                all_texts.append(text)
                new_files.append(file.name)
            else:
                st.warning(f"⚠️ {file.name} - No text found")
                
        except Exception as e:
            st.error(f"❌ Error in {file.name}: {str(e)}")
        
        progress_bar.progress((idx + 1) / total_files)
        gc.collect()  # Critical for mobile
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_texts:
        return False
    
    # Text splitting
    with st.spinner("Splitting text into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = []
        for text in all_texts:
            chunks.extend(text_splitter.split_text(text))
        gc.collect()
    
    # Embeddings (CPU-friendly for Streamlit Cloud)
    with st.spinner("Creating embeddings... (this may take a minute)"):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # ChromaDB with temp directory
        chroma_path = os.path.join(tempfile.gettempdir(), "chroma_db")
        
        # Clear old if exists (to avoid conflicts)
        if os.path.exists(chroma_path):
            import shutil
            shutil.rmtree(chroma_path, ignore_errors=True)
        
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=chroma_path
        )
        gc.collect()
    
    # Setup conversation chain
    with st.spinner("Setting up chat..."):
        llm = ChatGroq(
            api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY")),
            model_name="llama3-8b-8192",
            temperature=0.7
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            verbose=False
        )
    
    # Update session state
    st.session_state.vectorstore = vectorstore
    st.session_state.conversation = conversation
    st.session_state.processed_files.update(new_files)
    
    return True

# ============ HANDLE PROCESSING ============
if process_clicked and uploaded_files:
    success = process_documents(uploaded_files)
    if success:
        st.success(f"✅ Processed {len(uploaded_files)} file(s)! Ready to chat.")
        st.balloons()
    else:
        st.warning("No new files to process or all files empty.")
    st.rerun()

# ============ CHAT INTERFACE ============
if not st.session_state.processed_files:
    st.info("📤 Please upload PDF files from the sidebar and click 'Process Documents'")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation.invoke({"question": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Try reprocessing documents or check your API key.")

# ============ FOOTER ============
st.divider()
st.caption("🔒 Documents are processed in-memory only | No data is stored permanently")
