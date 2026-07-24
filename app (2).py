import os
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load OPENAI_API_KEY: from .env when running locally,
# from st.secrets when running on Streamlit Cloud
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error(
            "OPENAI_API_KEY가 설정되지 않았습니다.\n\n"
            "- 로컬 실행: 프로젝트 폴더에 .env 파일을 만들고 "
            "OPENAI_API_KEY=your_key 를 입력하세요.\n"
            "- Streamlit Cloud: 'Manage app' → Settings → Secrets에 "
            "OPENAI_API_KEY = \"your_key\" 를 입력하세요."
        )
        st.stop()

# Streamlit page config
st.set_page_config(page_title="Korean Document Q&A for International Students", layout="centered")
st.title("🎓 Korean University Document Q&A")
st.caption(
    "Upload a Korean-language document from your university "
    "(transcript, notice, guide, contract, etc.) and ask questions in English."
)

# 1. File upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Rebuild the vector store only when a new file is uploaded
    if "vectorstore" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Reading document and building search index..."):
            doc_bytes = uploaded_file.read()
            pdf_doc = fitz.open(stream=doc_bytes, filetype="pdf")

            text = ""
            for page in pdf_doc:
                text += page.get_text() + "\n\n"

            if not text.strip():
                st.error(
                    "No extractable text was found in this PDF. "
                    "It may be a scanned image — OCR is not supported yet."
                )
                st.stop()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = text_splitter.split_text(text)

            # Embeddings + FAISS vector store
            # text-embedding-3-large handles Korean text well
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            vectorstore = FAISS.from_texts(chunks, embeddings)

            st.session_state["vectorstore"] = vectorstore
            st.session_state["file_name"] = uploaded_file.name

        st.success("✅ Document processed! You can now ask questions below.")

    # 2. Question input
    query = st.text_input(
        "Ask a question about the document (in English):",
        placeholder="e.g. What is the deadline mentioned in this document?"
    )

    if query:
        with st.spinner("Generating answer..."):
            vectorstore = st.session_state["vectorstore"]

            # Retrieve relevant chunks (still in Korean)
            similar_docs = vectorstore.similarity_search_with_score(query, k=8)
            context = ""
            for doc, score in similar_docs:
                context += doc.page_content + "\n\n"

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            template = """You are a helpful assistant for international students studying in Korea.
The context below is extracted from a Korean-language university document.
Read and understand the Korean context carefully, then answer the user's
question in clear, natural English. If a specific term (e.g. a department
name, form name, or deadline) is important, you may include the original
Korean term in parentheses after the English translation.

If the answer cannot be found in the context, say so honestly instead of
guessing.

Context (in Korean):
{context}

Question (in English):
{question}

Answer in English:"""

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()

            response = chain.invoke({"context": context, "question": query})

            st.subheader("💡 Answer")
            st.write(response)
else:
    st.info("👆 Upload a PDF to get started.")
