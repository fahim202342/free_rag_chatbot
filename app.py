import streamlit as st
import os
import tempfile
import hashlib

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from operator import itemgetter

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

# ========== SESSION STATE INIT ==========
def init_session():
    defaults = {
        "uploaded_meta": {},
        "chunks": [],
        "chunk_embeddings": [],
        "embeddings": None,
        "has_docs": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        new_files = []
        all_chunks = []
        for file in uploaded_files:
            file_hash = hashlib.md5(file.getvalue()).hexdigest()
            if file_hash in st.session_state["uploaded_meta"]:
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

                all_chunks.extend(chunks)
                st.session_state["uploaded_meta"][file_hash] = file.name
                new_files.append(file.name)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
            finally:
                os.unlink(tmp_path)

        if all_chunks:
            # Embed all chunks
            texts = [c.page_content for c in all_chunks]
            chunk_embeddings = embeddings.embed_documents(texts)

            st.session_state["chunks"] = all_chunks
            st.session_state["chunk_embeddings"] = chunk_embeddings
            st.session_state["embeddings"] = embeddings
            st.session_state["has_docs"] = True
            st.success(f"✅ Uploaded: {', '.join(new_files)}")
            st.rerun()
        else:
            st.info("ℹ️ All files already uploaded.")

    if st.session_state["uploaded_meta"]:
        st.subheader("📁 Uploaded Files")
        for fh, fname in st.session_state["uploaded_meta"].items():
            st.text(f"• {fname}")

    if st.button("🗑️ Clear All Documents", type="secondary"):
        st.session_state["uploaded_meta"] = {}
        st.session_state["chunks"] = []
        st.session_state["chunk_embeddings"] = []
        st.session_state["embeddings"] = None
        st.session_state["has_docs"] = False
        st.success("Cleared!")
        st.rerun()

# ========== RETRIEVER ==========
MODEL = "llama-3.3-70b-versatile"
TEMP = 0.1
K_DOCS = 3

if not st.session_state["has_docs"]:
    st.info("📤 No documents uploaded yet. Please upload PDF files from the sidebar.")
    st.stop()

# Simple cosine similarity using pure Python
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_docs(query):
    """Retrieve top-k similar chunks using pure Python cosine similarity."""
    embeddings = st.session_state["embeddings"]
    chunks = st.session_state["chunks"]
    chunk_embeddings = st.session_state["chunk_embeddings"]

    if embeddings is None or not chunks or not chunk_embeddings:
        return []

    query_emb = embeddings.embed_query(query)

    scores = []
    for emb in chunk_embeddings:
        sim = cosine_similarity(query_emb, emb)
        scores.append(sim)

    # Get top K indices
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_k = indexed_scores[:K_DOCS]

    return [chunks[i] for i, _ in top_k]


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


def build_context(inputs):
    docs = retrieve_docs(inputs["question"])
    return format_docs(docs)


qa_rag_chain = (
    {
        "context": build_context,
        "question": itemgetter("question")
    }
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
                import pandas as pd
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
                config={"callbacks": [stream_handler, pm_handler]}
            )
        except Exception as e:
            st.error(f"Error: {e}")
