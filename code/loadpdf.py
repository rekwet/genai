# from langchain.document.loaders import PdfDocumentLoader
# from langchain.language.llm import LocalLanguageModel

from langchain.document_loaders import PyPDFLoader


def load_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    return pages


    
