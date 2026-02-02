import streamlit as st
from rag_engine import build_qa_from_pdf

st.set_page_config(page_title="Accolade PDF Chatbot", layout="wide")

st.title("📄 Accolade - PDF Chatbot (RAG)")
st.write("Upload a PDF and ask questions about it.")

# Initialize session state
if "qa" not in st.session_state:
    st.session_state.qa = None

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and not st.session_state.pdf_uploaded:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    with st.spinner("Processing PDF and building knowledge base..."):
        st.session_state.qa = build_qa_from_pdf("temp.pdf")
        st.session_state.pdf_uploaded = True

    st.success("Chatbot is ready!")

# Ask questions
if st.session_state.qa:
    question = st.text_input("Ask a question from the PDF")

    if st.button("Ask"):
        if question:
            with st.spinner("Thinking..."):
                answer = st.session_state.qa.run(question)
            st.markdown("### Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")
