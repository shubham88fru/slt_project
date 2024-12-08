import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationalBufferMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from template   import css, bot_template, user_template


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




def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embeddings=embeddings
    )
    return vectorstore



def get_coversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationalBufferMemory(memory_key="hist", return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def handle_userinput(user_question):
    response = st.session_state.conversation({
        "question": user_question
    })
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i %2 == 0:
            st.write(user_template.substitute("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.substitute("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="ChatPDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ChatPDF")
    user_question = st.text_input("Start asking a questions about your PDF file:")
    if user_question:
        handle_userinput(user_question)


    st.write(user_template.replace("{{MSG}}", "Hi"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "How can I help you?"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your PDF")
        pdf = st.file_uploader("Upload your PDF here to start chatting!")
        if st.button("Upload"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text([pdf])
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_coversation_chain(vectorstore)




if __name__ == "__main__":
    main()
