import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loader = PyPDFLoader("./data/telkom.pdf")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(loader)
db = FAISS.from_documents(documents, GPT4AllEmbeddings())