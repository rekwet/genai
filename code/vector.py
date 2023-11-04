from langchain.vectorstores import FAISS
from embed import embed_documents

def vectorize_documents(documents):
    embeddings = embed_documents(documents)
    db = FAISS.from_documents(docs, embeddings)

    query = "What did the president say about Ketanji Brown Jackson"
    docs = db.similarity_search(query)

    print(docs[0].page_content)
	