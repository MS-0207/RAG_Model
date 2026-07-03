# Apply MMR or LLM-based reranking
# rag_project/retrieval/reranker.py

from typing import List, Literal
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔐 Set your OpenAI API key

def rerank_documents(
    docs: List[Document],
    query: str,
    method: Literal["mmr", "cosine"] = "mmr",
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> List[Document]:
    """
    Rerank retrieved documents based on query relevance.

    :param docs: List of LangChain Documents
    :param query: The user query
    :param method: Reranking strategy ("mmr" or "cosine or multi-query")
    :param top_k: Number of documents to return
    :param lambda_param: MMR diversity-relevance trade-off
    :return: Reranked list of Documents
    """
    embeddings = OpenAIEmbeddings(openai_api_key="")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # query_emb = embeddings.embed_query(query)
    query_emb = np.array(embeddings.embed_query(query))
    doc_embs = embeddings.embed_documents([doc.page_content for doc in docs])

    if method == "mmr":
        selected_indices = maximal_marginal_relevance(
            query_embedding=query_emb,
            embedding_list=doc_embs,  # ✅ Correct
            k=top_k,
            lambda_mult=lambda_param
        )
        return [docs[i] for i in selected_indices]

    elif method == "cosine":
        sims = cosine_similarity([query_emb], doc_embs)[0]
        sorted_indices = np.argsort(sims)[::-1][:top_k]
        return [docs[i] for i in sorted_indices]

    else:
        raise ValueError("Unsupported reranking method. Use 'mmr' or 'cosine'.")


# 🧪 Example usage
if __name__ == "__main__":
    from retrieval.retrieve_topk import retrieve_top_k

    query = input("Enter Query:")
    docs = retrieve_top_k(query)

    reranked = rerank_documents(docs, query, method="mmr", top_k=5)
    for i, doc in enumerate(reranked, 1):
        print(f"\n🔹 Rank {i}:")
        print(doc.page_content[:300], "...")
