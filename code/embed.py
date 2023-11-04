from langchain.embeddings import GPT4AllEmbeddings

def embed_documents(documents):
    gpt4all_embd = GPT4AllEmbeddings()
    doc_results = gpt4all_embd.embed_documents(documents)
    return doc_results