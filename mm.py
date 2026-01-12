
# import streamlit as st
# from rag import process_urls, generate_answer

# st.title("Real Estate Research Tool")

# url1 = st.sidebar.text_input("URL 1")
# url2 = st.sidebar.text_input("URL 2")
# url3 = st.sidebar.text_input("URL 3")

# placeholder = st.empty()

# process_url_button = st.sidebar.button("Process URLs")
# if process_url_button:
#     urls = [url for url in (url1, url2, url3) if url!='']
#     if len(urls) == 0:
#         placeholder.text("You must provide at least one valid url")
#     else:
#         for status in process_urls(urls):
#             placeholder.text(status)

# query = placeholder.text_input("Question")
# if query:
#     try:
#         answer, sources = generate_answer(query)
#         st.header("Answer:")
#         st.write(answer)

#         if sources:
#             st.subheader("Sources:")
#             for source in sources.split("\n"):
#                 st.write(source)
#     except RuntimeError as e:
#         placeholder.text("You must process urls first")








import streamlit as st
from rag import process_documents, generate_answer




# @st.cache_resource
# def setup_nltk():
#     import nltk
#     nltk.download("punkt")
#     nltk.download("punkt_tab")

# setup_nltk()


st.set_page_config(page_title="Generic RAG Assistant", layout="wide")

st.title("📄 Generic RAG Document Assistant")
st.write("Ask questions from **URLs, PDFs, Docs, or TXT files** using GenAI 🚀")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Data Input")

input_type = st.sidebar.selectbox(
    "Select Data Source",
    ["URLs", "Upload Files"]
)

urls = []
uploaded_files = []

if input_type == "URLs":
    url1 = st.sidebar.text_input("URL 1")
    url2 = st.sidebar.text_input("URL 2")
    url3 = st.sidebar.text_input("URL 3")
    urls = [u for u in [url1, url2, url3] if u]

else:
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs / Docs / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

process_btn = st.sidebar.button("🚀 Process Data")

status_box = st.empty()

# ---------------- PROCESSING ---------------- #
if process_btn:
    if not urls and not uploaded_files:
        status_box.warning("Please provide URLs or upload files")
    else:
        for status in process_documents(
            urls=urls,
            files=uploaded_files
        ):
            status_box.info(status)

# ---------------- QUESTION ---------------- #
st.divider()
query = st.text_input("Ask a question from your data")

if query:
    try:
        answer, sources = generate_answer(query)

        st.subheader("✅ Answer")
        st.write(answer)

        if sources:
            st.subheader("📌 Sources")
            for src in sources.split("\n"):
                st.write(src)

    except RuntimeError:
        st.error("Please process data first.")
