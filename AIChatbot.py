from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    st.set_page_config(page_title="Provide your file and chat with me!")
    st.header("AI Chatbot")
#If variables are not in session then initialize them with null
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

#Sidebar creation using streamlit
    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload Your file here",type=['pdf'],accept_multiple_files=True)
        process = st.button("Process")
    if process:
        #Extracting text from uploaded file/files
        extractedText = file_text_extractor(uploaded_files)
        #Creating chunks from extracted text
        textChunks = text_chunking(extractedText)
        #Creating embeddings and storing in vector store
        vectorStore = embedding_and_vectorstore(textChunks)

        #Creating chain here. Memory is utilized here. Three important concepts here
        # 1. LLM - Main LLM to query
        # 2. Retriever - To get most relevant context (to user query) from vector store
        # 3. Memory - Memory of the conversation
        st.session_state.conversation = create_chain(vectorStore, os.getenv("openai_api_key"))
        st.session_state.processComplete = True
    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Write your query here....")
        if user_question:
            userinput(user_question)

# Parent function for text extraction - It will pass individual files to child function for text extraction
def file_text_extractor(uploadedFiles):
    text = ""
    for file in uploadedFiles:
        text += get_pdf_text(file)
    return text

#It will extract text from single file, page by page
def get_pdf_text(file):
    pdfReader = PdfReader(file)
    text = ""
    for page in pdfReader.pages:
        text += page.extract_text()
    return text

#Defining chunking configurations here.
# Separator - Main chunking condition. Chunk based on new line
# Chunk_size - Max size of chunk
# length_function - Using inbuild len function to calculate length for chunking
def text_chunking(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

#Function to create embedding and store in vector store
def embedding_and_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base

#Function to create chain. Memory is utilized to preserve context in conversation
def create_chain(vectorStore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain

#Function to handle user queries/questions
def userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))


if __name__ == '__main__':
    main()
