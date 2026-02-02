from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_text(text)


embeddings = OllamaEmbeddings(model="mistral")

db = FAISS.from_texts(chunks, embeddings)


llm = Ollama(model="mistral")

retriever = db.as_retriever()


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

while True:

    query = input("\nAsk (type exit to quit): ")

    if query.lower() == "exit":
        break

    answer = qa.run(query)

    print("\nAnswer:\n", answer)
