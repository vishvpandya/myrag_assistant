# from uuid import uuid4
# from dotenv import load_dotenv
# from pathlib import Path
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain_community.document_loaders import UnstructuredURLLoader
# # from langchain_community.document_loaders.unstructured import UnstructuredURLLoader

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# load_dotenv()

# # Constants
# CHUNK_SIZE = 1000
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
# COLLECTION_NAME = "real_estate"

# llm = None
# vector_store = None


# def initialize_components():
#     global llm, vector_store

#     if llm is None:
#         llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

#     if vector_store is None:
#         ef = HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL,
#             model_kwargs={"trust_remote_code": True}
#         )

#         vector_store = Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=ef,
#             persist_directory=str(VECTORSTORE_DIR)
#         )


# def process_urls(urls):
#     """
#     This function scraps data from a url and stores it in a vector db
#     :param urls: input urls
#     :return:
#     """
#     yield "Initializing Components"
#     initialize_components()

#     yield "Resetting vector store...✅"
#     vector_store.reset_collection()

#     yield "Loading data...✅"
#     loader = UnstructuredURLLoader(urls=urls)
#     data = loader.load()

#     yield "Splitting text into chunks...✅"
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", ".", " "],
#         chunk_size=CHUNK_SIZE
#     )
#     docs = text_splitter.split_documents(data)

#     yield "Add chunks to vector database...✅"
#     uuids = [str(uuid4()) for _ in range(len(docs))]
#     vector_store.add_documents(docs, ids=uuids)

#     yield "Done adding docs to vector database...✅"

# def generate_answer(query):
#     if not vector_store:
#         raise RuntimeError("Vector database is not initialized ")

#     chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
#     result = chain.invoke({"question": query}, return_only_outputs=True)
#     sources = result.get("sources", "")

#     return result['answer'], sources


# if __name__ == "__main__":
#     urls = [
#         "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
#         "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
#     ]

#     process_urls(urls)
#     answer, sources = generate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")
#     print(f"Answer: {answer}")
#     print(f"Sources: {sources}")










# @Author: Dhaval Patel
# Generic RAG Engine (URLs + Files)

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
