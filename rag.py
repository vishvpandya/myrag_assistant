from uuid import uuid4
from pathlib import Path
import os

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
)

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import nltk
import os

NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Download required NLTK resources (safe to run multiple times)
nltk.download("punkt", download_dir=NLTK_DATA_DIR, quiet=True)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR, quiet=True)


# ---------------- LOAD ENV ---------------- #
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("❌ GROQ_API_KEY not found. Add it to .env file.")

# ---------------- CONSTANTS ---------------- #
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).parent
VECTORSTORE_DIR = BASE_DIR / "resources/vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "generic_rag"

# ---------------- GLOBALS ---------------- #
llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500,
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )


def process_documents(urls=None, files=None):
    """
    Load URLs and/or files → chunk → embed → store in vector DB
    """
    initialize_components()

    yield "🔄 Resetting vector database..."
    vector_store.reset_collection()

    documents = []

    # -------- URL LOADING -------- #
    if urls:
        yield "🌐 Loading data from URLs..."
        try:
            loader = UnstructuredURLLoader(
                urls=urls,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            documents.extend(loader.load())
        except Exception:
            yield "⚠️ Failed to load one or more URLs."

    # -------- FILE LOADING -------- #
    if files:
        yield "📄 Loading uploaded files..."
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)

        for file in files:
            file_path = temp_dir / file.name
            file_path.write_bytes(file.read())

            if file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(str(file_path))
            else:
                loader = UnstructuredFileLoader(str(file_path))

            documents.extend(loader.load())

    if not documents:
        yield "❌ No documents found."
        yield "👉 Tip: Upload PDFs for documentation websites."
        return

    # -------- SPLITTING -------- #
    yield "✂️ Splitting documents into chunks..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # -------- STORE -------- #
    yield "🧠 Storing embeddings in vector database..."
    ids = [str(uuid4()) for _ in chunks]
    vector_store.add_documents(chunks, ids=ids)

    yield "✅ Data processing completed!"


def generate_answer(question: str):
    if vector_store is None:
        raise RuntimeError("Vector store not initialized.")

    prompt = PromptTemplate(
        template="""
You MUST answer the question using ONLY the context below.
If the answer is not in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15},
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    result = chain.invoke({"query": question})

    sources = set()
    for doc in result["source_documents"]:
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])

    return result["result"], list(sources)

