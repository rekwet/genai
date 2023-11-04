from langchain.text_splitter import RecursiveCharacterTextSplitter
from loadpdf import load_pdf

#TODO: add config file path
file = "sample.pdf"

def text_split(file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(load_pdf(file)) 
    return documents