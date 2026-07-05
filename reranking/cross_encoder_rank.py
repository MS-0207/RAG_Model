from typing import List
from langchain_core.documents import Document
from sentence_transformers.cross_encoder import CrossEncoder

def cross_encoder_rerank(
    query: str,
    docs: List[Document],
    top_k: int ,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
) -> List[Document]:
    if not docs:
        return []

    model = CrossEncoder(model_name)

    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]


    scores = model.predict(pairs)
    scored_docs = list(zip(docs, scores))


    scored_docs.sort(
        key=lambda x: float(x[1]),
        reverse=True
    )

    top_docs = [
        doc for doc, score in scored_docs[:top_k]
    ]

    return top_docs

