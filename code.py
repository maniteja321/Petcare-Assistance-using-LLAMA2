from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.google_generative_ai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.google_generative_ai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits the text into smaller chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Creates a vector store from the text chunks using Google Generative AI embeddings.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """
    Creates a conversational chain for question answering using Google Generative AI.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
    If the answer is not available in the provided context, just say "The answer is not available in the context."
    Do not provide any wrong or made-up information.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    """
    Processes a user question and returns the answer.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


# Additional functionality
def load_vector_store():
    """
    Loads an existing vector store from disk.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store


def search_vector_store(query, vector_store):
    """
    Searches the vector store for relevant documents based on the query.
    """
    docs = vector_store.similarity_search(query)
    return docs


def ask_question(question, vector_store):
    """
    Asks a question and retrieves the answer from the vector store.
    """
    docs = search_vector_store(question, vector_store)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]


# Additional functionality for handling PDF files
def process_pdf_files(pdf_files):
    """
    Processes a list of PDF files and creates a vector store.
    """
    combined_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(combined_text)
    get_vector_store(text_chunks)
    vector_store = load_vector_store()

    while True:
        user_query = input("Enter your question (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = ask_question(user_query, vector_store)
        print(f"Answer: {answer}")
