import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import gpt4all

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loader = PyPDFLoader("./data/machinelearning-lecture01.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(loader)
embeddings = GPT4AllEmbeddings()
db = FAISS.from_documents(documents, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())

