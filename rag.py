
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader,
    UnstructuredFileLoader
)

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

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
            max_tokens=500
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_documents(urls=None, files=None):
    """
    Process URLs and/or uploaded files and store embeddings in vector DB
    """
    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    documents = []

    # -------- Load URLs -------- #
    if urls:
        yield "Loading data from URLs...🌐"
        url_loader = UnstructuredURLLoader(urls=urls)
        documents.extend(url_loader.load())

    # -------- Load Files -------- #
    if files:
        yield "Loading data from uploaded files...📄"
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)

        for file in files:
            file_path = temp_dir / file.name
            file_path.write_bytes(file.read())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(str(file_path))
            else:
                loader = UnstructuredFileLoader(str(file_path))

            documents.extend(loader.load())

    if not documents:
        yield "No documents found ❌"
        return

    # -------- Chunking -------- #
    yield "Splitting text into chunks...✂️"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    # -------- Store Embeddings -------- #
    yield "Adding chunks to vector database...🧠"
    ids = [str(uuid4()) for _ in chunks]
    vector_store.add_documents(chunks, ids=ids)

    yield "✅ Processing completed successfully!"


def generate_answer(query: str):
    if vector_store is None:
        raise RuntimeError("Vector database not initialized")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    result = chain.invoke({"question": query}, return_only_outputs=True)

    return result["answer"], result.get("sources", "")

