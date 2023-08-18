from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import openai,HuggingFaceHub
from langchain.chat_models import ChatOpenAI

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_split = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_split.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_userinput(user_question, conversation_chain):
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            print("User:", message.content)
        else:
            print("Bot:", message.content)

def main():
    load_dotenv()
    pdf_folder = 'docs'
    
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]
    pdf_paths = [os.path.join(pdf_folder, file) for file in pdf_files]
    print(pdf_paths)

    # get pdf text
    raw_text = get_pdf_text(pdf_paths)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    conversation_chain = conversation_chain(vectorstore)

    while True:
        user_question = input("Ask a question about your documents (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        handle_userinput(user_question, conversation_chain)

if __name__ == '__main__':
    main()
