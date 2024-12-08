import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


def get_pdf_text(pdf):
    text = ""
    for p in pdf:
        pdf_reader = PdfReader(p)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_slitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_slitter.split_text(raw_text)
    return chunks




def main():
    load_dotenv()
    st.set_page_config(page_title="ChatPDF", page_icon=":books:")
    st.header("ChatPDF")
    st.text_input("Start asking a questions about your PDF file:")

    with st.sidebar:
        st.subheader("Your PDF")
        pdf = st.file_uploader("Upload your PDF here to start chatting!")
        if st.button("Upload"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text([pdf])
                text_chunks = get_text_chunks(raw_text)


if __name__ == "__main__":
    main()
