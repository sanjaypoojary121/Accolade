from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


def build_qa_from_pdf(pdf_path):
    # 1. Read PDF
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # 2. Split text (optimized)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    # 3. Embeddings
    embeddings = OllamaEmbeddings(model="mistral")

    # 4. Vector DB
    db = FAISS.from_texts(chunks, embeddings)

    # 5. Retriever (LIMITED)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # 6. LLM (QUANTIZED + LIMITED)
    llm = Ollama(
        model="mistral:7b-instruct-q4_K_M",
        temperature=0.1,
        num_predict=256
    )

    # 7. RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        verbose=False
    )

    return qa
