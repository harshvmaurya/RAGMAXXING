import hashlib
import html
import os
from io import BytesIO
from pathlib import Path
from textwrap import dedent

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError


load_dotenv()
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only the supplied document context. "
            "If the answer is not in the context, say you do not know.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@st.cache_resource
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_groq_api_key() -> str | None:
    try:
        secret = st.secrets.get("GROQ_API_KEY")
    except FileNotFoundError:
        secret = None
    return secret or os.getenv("GROQ_API_KEY")


def uploaded_files_fingerprint(files: list) -> str:
    digest = hashlib.sha256()
    for uploaded_file in files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def load_uploaded_documents(files: list) -> list[Document]:
    documents = []
    for uploaded_file in files:
        try:
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        except PdfReadError as error:
            st.error(f"Could not read {uploaded_file.name}: {error}")
            continue

        extracted_pages = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_number,
                        },
                    )
                )
                extracted_pages += 1

        if not extracted_pages:
            st.warning(
                f"No selectable text was found in {uploaded_file.name}. "
                "Scanned PDFs need OCR before they can be searched."
            )
    return documents


def format_source(document: Document) -> str:
    source = Path(str(document.metadata.get("source", "PDF"))).name
    page = document.metadata.get("page")
    return f"{source}, page {page}" if page is not None else source


st.set_page_config(page_title="RAGMAXXING", page_icon=":material/menu_book:")
st.markdown(
    """
    <style>
    :root {
        --rag-ink: #171717;
        --rag-cream: #f7f3e9;
        --rag-yellow: #ffde59;
        --rag-cyan: #b9edfa;
        --rag-white: #ffffff;
        --rag-surface: #e7ecee;
        --rag-gray: #555555;
        --rag-border: 2px solid var(--rag-ink);
        --rag-shadow: 5px 5px 0 var(--rag-ink);
    }

    html,
    body,
    .stApp {
        font-family: Inter, "Segoe UI", Arial, sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--rag-cream);
        color: var(--rag-ink);
    }

    [data-testid="stHeader"] {
        background: var(--rag-cream);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 900px;
        padding: 1.6rem 1.5rem 3.5rem;
    }

    .ragmaxxing-brand {
        margin: 0 0 1.55rem;
        padding: 0.95rem 1.35rem;
        border: var(--rag-border);
        border-radius: 6px;
        background: var(--rag-yellow);
        box-shadow: var(--rag-shadow);
    }

    .ragmaxxing-brand p {
        margin: 0;
        color: var(--rag-ink);
    }

    .ragmaxxing-brand .brand-eyebrow {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .ragmaxxing-brand .brand-title {
        margin-top: 0.12rem;
        font-family: "Space Grotesk", Inter, "Arial Black", sans-serif;
        font-size: clamp(2rem, 5vw, 2.65rem);
        font-weight: 900;
        letter-spacing: -0.045em;
        line-height: 1.08;
    }

    .ragmaxxing-brand .brand-description {
        margin-top: 0.35rem;
        font-size: 0.96rem;
        font-weight: 500;
    }

    .section-intro {
        margin: 0 0 0.9rem;
    }

    .section-intro .section-kicker {
        margin: 0 0 0.25rem;
        color: var(--rag-gray);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .section-intro h2 {
        margin: 0;
        color: var(--rag-ink);
        font-size: clamp(1.2rem, 3vw, 1.5rem);
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.2;
    }

    .section-intro p {
        margin: 0.35rem 0 0;
        color: var(--rag-gray);
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stFileUploader"] {
        color: var(--rag-ink);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed var(--rag-ink);
        border-radius: 6px;
        background: var(--rag-cyan);
        box-shadow: 4px 4px 0 var(--rag-ink);
        transition: background-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploaderDropzone"]:focus-within {
        background: #a8e7f5;
        box-shadow: 5px 5px 0 var(--rag-ink);
    }

    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploader"] small {
        color: var(--rag-ink) !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        border: var(--rag-border);
        border-radius: 4px;
        background: var(--rag-white);
        box-shadow: 3px 3px 0 var(--rag-ink);
        color: var(--rag-ink) !important;
        font-weight: 700;
        transition: transform 100ms ease, box-shadow 100ms ease, background-color 100ms ease;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: var(--rag-ink);
        background: var(--rag-yellow);
        color: var(--rag-ink) !important;
        transform: translate(1px, 1px);
        box-shadow: 2px 2px 0 var(--rag-ink);
    }

    [data-testid="stFileUploaderDropzone"] button:active {
        transform: translate(2px, 2px);
        box-shadow: 1px 1px 0 var(--rag-ink);
    }

    div.stButton > button {
        min-height: 2.75rem;
        border: var(--rag-border);
        border-radius: 5px;
        background: var(--rag-yellow);
        box-shadow: 4px 4px 0 var(--rag-ink);
        color: var(--rag-ink);
        font-weight: 750;
        transition: transform 100ms ease, box-shadow 100ms ease, background-color 100ms ease;
    }

    div.stButton > button:hover {
        border-color: var(--rag-ink);
        background: #ffcf24;
        color: var(--rag-ink);
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0 var(--rag-ink);
    }

    div.stButton > button:active {
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0 var(--rag-ink);
    }

    .stApp input,
    .stApp textarea {
        border-color: var(--rag-ink);
        color: var(--rag-ink);
    }

    .stApp input:focus,
    .stApp textarea:focus,
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--rag-ink);
        box-shadow: 0 0 0 2px var(--rag-cyan), 3px 3px 0 var(--rag-ink);
    }

    .stApp input::placeholder,
    .stApp textarea::placeholder {
        color: var(--rag-gray);
        opacity: 1;
    }

    [data-testid="stChatMessage"] {
        margin-bottom: 0.8rem;
        border: var(--rag-border);
        border-radius: 6px;
        background: var(--rag-white);
        box-shadow: 4px 4px 0 var(--rag-ink);
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: var(--rag-ink);
        line-height: 1.65;
    }

    [data-testid="stChatInput"] {
        border: var(--rag-border);
        border-radius: 6px;
        background: var(--rag-white);
        box-shadow: 3px 3px 0 var(--rag-ink);
        transition: box-shadow 120ms ease;
    }

    [data-testid="stChatInput"] textarea {
        background: var(--rag-white);
        color: var(--rag-ink);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--rag-gray);
        opacity: 1;
    }

    [data-testid="stChatMessage"] a {
        color: var(--rag-ink);
        font-weight: 700;
        text-decoration-thickness: 1px;
        text-underline-offset: 2px;
    }

    [data-testid="stAlert"] {
        border: var(--rag-border);
        border-radius: 5px;
        background: var(--rag-white);
        box-shadow: 3px 3px 0 var(--rag-ink);
    }

    [data-testid="stAlert"] p {
        color: var(--rag-ink);
    }

    .empty-state {
        margin: 0.5rem 0 1.4rem;
        padding: 1rem 1.1rem;
        border: var(--rag-border);
        border-radius: 5px;
        background: var(--rag-white);
        box-shadow: 3px 3px 0 var(--rag-ink);
    }

    .empty-state strong {
        display: block;
        margin-bottom: 0.2rem;
        color: var(--rag-ink);
        font-size: 0.98rem;
    }

    .empty-state p {
        margin: 0;
        color: var(--rag-gray);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .document-library {
        margin: 1.35rem 0 1.5rem;
    }

    .document-library-heading {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .document-library-heading h3 {
        margin: 0;
        color: var(--rag-ink);
        font-size: 1.1rem;
        font-weight: 800;
    }

    .document-library-heading span {
        color: var(--rag-gray);
        font-size: 0.85rem;
    }

    .document-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.65rem;
        padding: 0.8rem 0.95rem;
        border: var(--rag-border);
        border-radius: 5px;
        background: var(--rag-white);
        box-shadow: 3px 3px 0 var(--rag-ink);
    }

    .document-name {
        overflow-wrap: anywhere;
        color: var(--rag-ink);
        font-size: 0.95rem;
        font-weight: 700;
    }

    .document-meta {
        margin-top: 0.2rem;
        color: var(--rag-gray);
        font-size: 0.82rem;
    }

    .document-status {
        flex: 0 0 auto;
        padding: 0.25rem 0.45rem;
        border: 1px solid var(--rag-ink);
        border-radius: 3px;
        background: var(--rag-cyan);
        color: var(--rag-ink);
        font-size: 0.76rem;
        font-weight: 700;
    }

    .document-status.is-indexed {
        background: var(--rag-yellow);
    }

    :focus-visible {
        outline: 3px solid var(--rag-ink) !important;
        outline-offset: 3px !important;
    }

    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.25rem 1rem 3rem;
        }

        .ragmaxxing-brand {
            margin-bottom: 1.4rem;
            padding: 1rem 1.1rem;
        }

        .document-row {
            align-items: flex-start;
        }

        .document-status {
            white-space: nowrap;
        }
    }
    </style>
    <div class="ragmaxxing-brand">
        <p class="brand-eyebrow">Your personal document library</p>
        <p class="brand-title">RAGMAXXING</p>
        <p class="brand-description">Upload. Search. Understand.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_vectorstore" not in st.session_state:
    st.session_state.uploaded_vectorstore = None
if "indexed_fingerprint" not in st.session_state:
    st.session_state.indexed_fingerprint = None
vectorstore = None
current_fingerprint = None

st.markdown(
    """
    <div class="section-intro">
        <p class="section-kicker">01 / Build your library</p>
        <h2>Start with your sources</h2>
        <p>Add text-based PDFs to create a private, searchable workspace for this session.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Choose one or more text-based PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="PDFs are indexed in memory for this session.",
)

if uploaded_files:
    current_fingerprint = uploaded_files_fingerprint(uploaded_files)
    document_library = st.empty()

    if st.button("Index PDFs", type="primary"):
        with st.spinner("Reading PDFs and creating the search index..."):
            documents = load_uploaded_documents(uploaded_files)
            if documents:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                chunks = splitter.split_documents(documents)
                st.session_state.uploaded_vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=load_embeddings(),
                )
                st.session_state.indexed_fingerprint = current_fingerprint
                st.session_state.messages = []
                st.success(
                    f"Indexed {len(uploaded_files)} PDF(s) into "
                    f"{len(chunks)} searchable chunks."
                )

    is_indexed = st.session_state.indexed_fingerprint == current_fingerprint
    library_markup = [
        dedent(f"""
        <section class="document-library" aria-label="Uploaded documents">
            <div class="document-library-heading">
                <h3>Document library</h3>
                <span>{len(uploaded_files)} PDF{"s" if len(uploaded_files) != 1 else ""}</span>
            </div>
        """)
    ]
    for uploaded_file in uploaded_files:
        size_mb = uploaded_file.size / (1024 * 1024)
        filename = html.escape(uploaded_file.name)
        status = "Indexed" if is_indexed else "Ready to index"
        status_class = "is-indexed" if is_indexed else ""
        library_markup.append(
            dedent(f"""
            <div class="document-row">
                <div>
                    <div class="document-name">{filename}</div>
                    <div class="document-meta">PDF | {size_mb:.2f} MB</div>
                </div>
                <span class="document-status {status_class}">{status}</span>
            </div>
            """)
        )
    library_markup.append(
        "</section><p class=\"document-meta\">To remove a PDF, use its remove control in the upload area, then index the updated selection.</p>"
    )
    document_library.markdown(
        "\n".join(library_markup),
        unsafe_allow_html=True,
    )

    if (
        st.session_state.uploaded_vectorstore is not None
        and st.session_state.indexed_fingerprint == current_fingerprint
    ):
        vectorstore = st.session_state.uploaded_vectorstore
    else:
        st.info("Index the selected PDFs before asking questions.")
else:
    st.markdown(
        """
        <div class="empty-state">
            <strong>Your library starts with a PDF</strong>
            <p>Choose or drop one or more text-based PDFs above. They stay in this session and are not saved as a permanent library.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if vectorstore is not None:
    st.markdown(
        """
        <div class="section-intro">
            <p class="section-kicker">02 / Explore your library</p>
            <h2>Ask your documents</h2>
            <p>Answers use the indexed text and include source pages so you can verify the details.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("Sources: " + "; ".join(message["sources"]))

    question = st.chat_input("Ask a question about the selected documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            api_key = get_groq_api_key()
            if not api_key:
                st.error(
                    "GROQ_API_KEY is missing. Set it in your local .env file "
                    "or in the deployed app's Streamlit secrets."
                )
            else:
                with st.spinner("Searching the documents and generating an answer..."):
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 4,
                            "fetch_k": 10,
                            "lambda_mult": 0.5,
                        },
                    )
                    relevant_documents = retriever.invoke(question)
                    if not relevant_documents:
                        answer = "I could not find relevant text in the selected documents."
                        sources = []
                    else:
                        context = "\n\n".join(
                            document.page_content for document in relevant_documents
                        )
                        response = ChatGroq(
                            model="openai/gpt-oss-120b",
                            api_key=api_key,
                        ).invoke(
                            CHAT_PROMPT.invoke(
                                {"context": context, "question": question}
                            )
                        )
                        answer = str(response.content)
                        sources = list(
                            dict.fromkeys(
                                format_source(document)
                                for document in relevant_documents
                            )
                        )

                st.markdown(answer)
                if sources:
                    st.caption("Sources: " + "; ".join(sources))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
