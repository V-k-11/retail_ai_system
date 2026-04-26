from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def build_vector_store(docs: list[str]):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(docs, embeddings)


def query_rag(store, question: str, k: int = 4):
    return store.similarity_search(question, k=k)

