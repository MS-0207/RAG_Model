# retrieval/hybrid_retriever.py

from typing import List
import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_DIR = r"C:\Users\msdha\PycharmProjects\RAG_Project\RAG\embeddings\cache"

def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    db = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


def get_all_docs_from_faiss(db) -> List[Document]:
    return list(db.docstore._dict.values())


def bm25_retrieval(
    query: str,
    docs: List[Document],
    top_k: int = 10
) -> List[Document]:

    if not docs:
        return []

    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in docs
    ]

    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc for doc, score in scored_docs[:top_k]
    ]


def vector_mmr_retrieval(
    query: str,
    db,
    top_k: int = 10,
    fetch_k: int = 30,
    lambda_mult: float = 0.5
) -> List[Document]:

    mmr_docs = db.max_marginal_relevance_search(
        query=query,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

    return mmr_docs

def merge_and_deduplicate(
    bm25_docs: List[Document],
    vector_docs: List[Document]
) -> List[Document]:

    unique_docs = {}

    merged_docs = bm25_docs + vector_docs

    for doc in merged_docs:
        source = doc.metadata.get("source", "")
        content = doc.page_content

        unique_key = f"{source}_{hash(content)}"

        if unique_key not in unique_docs:
            unique_docs[unique_key] = doc

    return list(unique_docs.values())


def hybrid_retrieval(
    query: str,
    bm25_top_k: int = 10,
    vector_top_k: int = 10,
    fetch_k: int = 30,
    lambda_mult: float = 0.5
) -> List[Document]:

    db = load_vector_store()

    all_docs = get_all_docs_from_faiss(db)

    bm25_docs = bm25_retrieval(
        query=query,
        docs=all_docs,
        top_k=bm25_top_k
    )

    vector_docs = vector_mmr_retrieval(
        query=query,
        db=db,
        top_k=vector_top_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

    final_docs = merge_and_deduplicate(
        bm25_docs=bm25_docs,
        vector_docs=vector_docs
    )

    return final_docs

