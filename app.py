import streamlit as st
import os
import tempfile
import hashlib
import json

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
import pandas as pd
import numpy as np

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
if "uploaded_meta" not in st.session_state:
    st.session_state["uploaded_meta"] = {}

if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None

if "has_docs" not in st.session_state:
    st.session_state["has_docs"] = False

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

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
            # Try FAISS first (pure in-memory)
            try:
                from langchain_community.vectorstores import FAISS
                vectordb = FAISS.from_documents(all_chunks, embeddings)
                st.session_state["vectordb"] = vectordb
                st.session_state["has_docs"] = True
                st.session_state["use_faiss"] = True
                st.success(f"✅ Uploaded: {', '.join(new_files)}")
                st.rerun()
            except Exception as e:
                # Fallback: simple similarity search
                st.session_state["chunks"] = all_chunks
                st.session_state["embeddings"] = embeddings
                st.session_state["has_docs"] = True
                st.session_state["use_faiss"] = False
                st.success(f"✅ Uploaded: {', '.join(new_files)} (fallback mode)")
                st.rerun()
        else:
            st.info("ℹ️ All files already uploaded.")

    if st.session_state["uploaded_meta"]:
        st.subheader("📁 Uploaded Files")
        for fh, fname in st.session_state["uploaded_meta"].items():
            st.text(f"• {fname}")

    if st.button("🗑️ Clear All Documents", type="secondary"):
        st.session_state["uploaded_meta"] = {}
        st.session_state["vectordb"] = None
        st.session_state["has_docs"] = False
        st.session_state["chunks"] = []
        st.session_state["use_faiss"] = True
        st.success("Cleared!")
        st.rerun()

# ========== RETRIEVER ==========
MODEL = "llama-3.3-70b-versatile"
TEMP = 0.1
K_DOCS = 3

if not st.session_state["has_docs"]:
    st.info("📤 No documents uploaded yet. Please upload PDF files from the sidebar.")
    st.stop()

# Build retriever from session state
if st.session_state.get("use_faiss", True) and st.session_state["vectordb"] is not None:
    retriever = st.session_state["vectordb"].as_retriever(search_kwargs={"k": K_DOCS})
else:
    # Fallback: simple similarity search using embeddings
    from sklearn.metrics.pairwise import cosine_similarity

    class SimpleRetriever:
        def __init__(self, chunks, embeddings, k):
            self.chunks = chunks
            self.embeddings = embeddings
            self.k = k
            texts = [c.page_content for c in chunks]
            self.chunk_embeddings = self.embeddings.embed_documents(texts)

        def get_relevant_documents(self, query):
            query_emb = self.embeddings.embed_query(query)
            similarities = cosine_similarity([query_emb], self.chunk_embeddings)[0]
            top_k_idx = np.argsort(similarities)[::-1][:self.k]
            results = []
            for i in top_k_idx:
                doc = self.chunks[i]
                # Ensure we return proper Document objects
                if isinstance(doc, Document):
                    results.append(doc)
                else:
                    results.append(Document(page_content=str(doc), metadata={}))
            return results

        def invoke(self, query):
            return self.get_relevant_documents(query)

    retriever = SimpleRetriever(
        st.session_state.get("chunks", []),
        st.session_state.get("embeddings"),
        K_DOCS
    )

# ========== LLM & CHAIN ==========
llm = ChatGroq(model_name=MODEL, temperature=TEMP, streaming=True)

qa_template = """You are a helpful AI assistant. Use ONLY the following context to answer the question. If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question: {question}

Answer:"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)


def format_docs(docs):
    if isinstance(docs, str):
        return docs
    return "\n\n".join([d.page_content for d in docs])


def retrieve_docs(inputs):
    """Retrieve documents and ensure we return a list of Document objects."""
    query = inputs["question"]
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
    else:
        docs = retriever.get_relevant_documents(query)
    return docs


qa_rag_chain = (
    {
        "context": RunnableLambda(retrieve_docs) | format_docs,
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
