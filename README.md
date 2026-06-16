# Free RAG Chatbot

In this project, I have built a completely free RAG (Retrieval-Augmented Generation) chatbot that can talk to your PDF documents. I am using the Groq API free tier to power the LLM responses and HuggingFace embeddings for local vector search.

---

## What is RAG and How It Works

RAG stands for Retrieval-Augmented Generation. It is a technique where the AI first searches through your documents to find relevant information, and then uses that information to generate accurate answers. This reduces hallucination and makes the responses fact-based.

Here is the step-by-step flow of how this chatbot works:

1. You upload PDF files through the sidebar
2. The PDFs are parsed and the text is extracted using PyMuPDF
3. The text is split into smaller chunks using RecursiveCharacterTextSplitter
4. Each chunk is converted into a vector embedding using HuggingFace all-MiniLM-L6-v2 model
5. These embeddings are stored in ChromaDB, a vector database
6. When you ask a question, the chatbot converts your question into an embedding too
7. It searches ChromaDB for the most similar document chunks
8. The retrieved chunks are sent to Groq's Llama 3.3 70B model along with your question
9. The LLM generates a precise answer based only on the retrieved context
10. The source document name and page number are displayed below the answer

---

## Features

- Completely Free: Uses Groq API free tier, no credit card required
- Local Embeddings: HuggingFace model runs on your machine, no API key needed for embeddings
- Multi-PDF Support: Upload and chat with multiple PDFs at the same time
- Source Tracking: See exactly which document and page the answer came from
- Streaming Response: Watch the answer being typed in real-time
- Custom System Prompt: Change how the AI behaves from the sidebar
- Persistent Storage: Uploaded documents stay saved between sessions
- Safe Cleanup: One-click option to clear all documents and reset the database

---

## Technologies Used

- Python 3.9 or higher
- Streamlit for the web interface
- LangChain for the RAG pipeline
- Groq API for fast LLM inference (Llama 3.3 70B)
- HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
- ChromaDB for vector storage and similarity search
- PyMuPDF for PDF parsing and text extraction
- Pandas for displaying source metadata in tables
- python-dotenv for managing environment variables

---

## How to Set Up the API Key

You need a Groq API key to use this chatbot. Here is how to get one:

1. Go to https://console.groq.com and create a free account
2. Once logged in, click on API Keys in the left sidebar
3. Click Create API Key and give it a name
4. Copy the generated key (it starts with gsk_)
5. Create a file named .env in your project folder
6. Add this line to the file: GROQ_API_KEY=your_copied_key_here
7. Save the file. Do not share this file with anyone

That is it. Your API key is now set up.

---

## Installation and Running

First, make sure you have Python 3.9 or higher installed on your system.

Step 1: Clone the repository or download the project files to your computer.

Step 2: Open a terminal or command prompt and navigate to the project folder.

Step 3: Create a virtual environment (this keeps dependencies isolated):

    python -m venv venv

Step 4: Activate the virtual environment.

For Windows:

    venv\Scripts\activate

For macOS or Linux:

    source venv/bin/activate

Step 5: Install all the required packages:

    pip install -r requirements.txt

This will download and install Streamlit, LangChain, ChromaDB, PyMuPDF, Pandas, python-dotenv, and all other dependencies.

Step 6: Set up your Groq API key as described in the section above.

Step 7: Run the application:

    streamlit run app.py

Your browser should automatically open with the chatbot interface. If it does not, copy the local URL shown in the terminal and paste it into your browser.

---

## Project Files

The project contains the following files:

app.py: This is the main application file. It contains the Streamlit interface, the RAG pipeline, the document upload logic, the chat interface, and the system prompt editor.

requirements.txt: This file lists all the Python packages needed to run the project. You can install them all at once using pip.

.env: This file stores your Groq API key. It is created by you and is not included in the Git repository for security reasons.

.env.example: This is a template file that shows you how to create the .env file. It contains the format but no real API key.

.gitignore: This file tells Git which files and folders to ignore. It ensures that sensitive files like .env and auto-generated folders like chroma_db are not uploaded to GitHub.

chroma_db: This folder is created automatically when you first upload a PDF. It stores the vector embeddings of your documents. You do not need to create or manage this folder manually.

system_prompt.txt: This file is created automatically when you save a custom system prompt. It stores your prompt so it persists between sessions.

uploaded_files_meta.json: This file is created automatically to keep track of which PDFs have been uploaded. It prevents duplicate uploads.

---

## How to Customize the Chatbot

The sidebar in the application gives you several options:

Upload Documents: Click the file uploader and select one or more PDF files. The chatbot will process them automatically. Duplicate files are skipped.

Edit System Prompt: Expand this section to change how the AI behaves. The system prompt tells the AI what personality or rules to follow. For example, you can make it act like a teacher, a lawyer, or a concise assistant.

Here are some example prompts you can try:

Professional Mode:
You are a professional research assistant. Answer questions based strictly on the provided documents. Use formal language and cite sources when possible.

Concise Mode:
You are a concise AI. Answer in 2 to 3 sentences maximum using only the context provided.

Teacher Mode:
You are a patient teacher. Explain answers simply as if teaching a student. Use examples from the documents whenever possible.

Clear All Documents: Click this button to delete all uploaded files and reset the vector database. This is useful when you want to start fresh with new documents.

---

## Groq Free Tier Limits

Groq offers a generous free tier for developers. Here are the current limits:

- 20 requests per minute
- 1,000,000 tokens per day
- 6,000 requests per day

These limits are more than enough for personal use and small projects.

---

## Common Issues and How to Fix Them

Problem: ModuleNotFoundError when running the app
Solution: This means some packages are missing. Run pip install -r requirements.txt again to install all dependencies.

Problem: API Key invalid or missing error
Solution: Make sure you have created the .env file in the project folder and added your Groq API key correctly. The key should start with gsk_. If the key is old, generate a new one from the Groq console.

Problem: PermissionError on Windows when clearing documents
Solution: This happens because Windows locks the SQLite database file while it is in use. Simply restart the application and try again. The app includes safe handling for this situation.

Problem: CUDA out of memory warning
Solution: This is not a problem for this project. The embedding model runs on CPU by default, so you do not need a GPU.

---

## Important Notes

- The embedding model all-MiniLM-L6-v2 is approximately 400MB in size. It downloads automatically the first time you run the app. No manual setup is needed.
- You do not need an API key for the embeddings. They run completely locally on your machine.
- Uploaded documents are stored in the chroma_db folder on your computer. They remain there until you manually clear them.
- The .env file contains your private API key. It is automatically ignored by Git, so it will never be uploaded to GitHub by accident.
- The chatbot only answers based on the uploaded documents. If the answer is not found in the documents, it will tell you that it does not have enough information.

---

## License

This project is released under the MIT License. You are free to use, modify, and distribute it as you wish.

---

## Credits and Acknowledgments

- LangChain for providing the RAG framework
- Groq for offering fast and free LLM inference
- Streamlit for making web app development simple
- HuggingFace for open-source embedding models
- ChromaDB for the open-source vector database

---

If you find this project helpful, please consider giving it a star on GitHub.