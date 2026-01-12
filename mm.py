import streamlit as st
from rag import process_documents, generate_answer

st.set_page_config(page_title="Generic RAG Assistant", layout="wide")

st.title("📄 Generic RAG Document Assistant")
st.write("Ask questions from URLs or Documents using GenAI 🚀")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Data Input")

input_type = st.sidebar.selectbox(
    "Select Data Source",
    ["URLs", "Upload Documents"]
)

urls = []
files = []

if input_type == "URLs":
    url1 = st.sidebar.text_input("URL 1")
    url2 = st.sidebar.text_input("URL 2")
    url3 = st.sidebar.text_input("URL 3")
    urls = [u for u in [url1, url2, url3] if u]

else:
    files = st.sidebar.file_uploader(
        "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

process_btn = st.sidebar.button("🚀 Process Data")

status_box = st.empty()

if process_btn:
    if not urls and not files:
        status_box.warning("Please provide URLs or upload documents.")
    else:
        for status in process_documents(urls=urls, files=files):
            status_box.info(status)

# ---------------- QUESTION ---------------- #
st.divider()
question = st.text_input("Ask a question from your data")

if question:
    try:
        answer, sources = generate_answer(question)

        st.subheader("✅ Answer")
        st.write(answer)

        if sources:
            st.subheader("📌 Sources")
            for src in sources:
                st.write(src)

    except RuntimeError as e:
        st.error(str(e))
