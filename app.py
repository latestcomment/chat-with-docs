import streamlit as st

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import os
from dotenv import load_dotenv

from html_templates import chat_css, user_template, bot_template

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST"),
        port=int(os.getenv("CHROMA_PORT")),
        settings=Settings()
    )
collection_name = os.getenv("CHROMA_COLLECTION_NAME")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    vectorstore.add_texts(text_chunks)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = st.session_state.model
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot",
                       page_icon=":nerd_face:")
    st.write(chat_css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "model" not in st.session_state:
        st.session_state.model = Ollama(model="llama3.1:latest")

    st.header("Chat with documents :bookmark_tabs:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question=user_question)

    with st.sidebar:
        st.subheader("Your model")
        option = st.selectbox("Select model",
                              ("None","llama","GPT"))
        if option == "None":
            st.write("Select model")
        elif option == "GPT":
            st.session_state.model = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        elif option == 'Llama':
            st.session_state.model = Ollama(model="llama3.1:latest")

        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()