import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from template import css, bot_template, user_template


def extract_pdf_content(uploaded_pdfs):
    combined_text = ""
    for pdf_file in uploaded_pdfs:
        pdf_parser = PdfReader(pdf_file)
        for page in pdf_parser.pages:
            combined_text += page.extract_text()
    return combined_text


def split_text_to_chunks(document_text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(document_text)


def create_vector_store(chunks):
    text_embeddings = OpenAIEmbeddings()
    # Alternative embeddings can be used as needed
    vector_storage = FAISS.from_texts(
        texts=chunks,
        embeddings=text_embeddings
    )
    return vector_storage


def initialize_conversation_chain(vector_storage):
    language_model = ChatOpenAI()
    buffer_memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=language_model,
        retriever=vector_storage.as_retriever(),
        memory=buffer_memory
    )
    return retrieval_chain


def process_user_input(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_log = response['chat_history']
    for index, chat_message in enumerate(st.session_state.chat_log):
        if index % 2 == 0:  # User's message
            st.write(user_template.substitute("{{MSG}}", chat_message.content), unsafe_allow_html=True)
        else:  # Bot's response
            st.write(bot_template.substitute("{{MSG}}", chat_message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="ChatPDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = None

    st.header("PDF Chat Assistant")
    user_query = st.text_input("Ask a question about your PDF:")
    if user_query:
        process_user_input(user_query)

    st.write(user_template.replace("{{MSG}}", "Hi"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "How can I assist you?"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Upload Your PDF")
        uploaded_pdf = st.file_uploader("Upload your PDF to begin.")
        if st.button("Process PDF"):
            with st.spinner("Analyzing the file..."):
                pdf_text = extract_pdf_content([uploaded_pdf])
                text_chunks = split_text_to_chunks(pdf_text)
                vector_storage = create_vector_store(text_chunks)
                st.session_state.conversation = initialize_conversation_chain(vector_storage)


if __name__ == "__main__":
    main()
