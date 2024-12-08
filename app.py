import streamlit as st


def main():
    st.set_page_config(page_title="ChatPDF", page_icon=":books:")
    st.header("ChatPDF")
    st.text_input("Start asking a questions about your PDF file:")

    with st.sidebar:
        st.subheader("Your PDF")
        st.file_uploader("Upload your PDF here to start chatting!")
        st.button("Upload")


if __name__ == "__main__":
    main()
