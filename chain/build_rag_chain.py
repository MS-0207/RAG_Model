# Connect retriever and LLM
# rag_project/chain/qa_chain.py

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from retrieval.retrieve_topk import retrieve_top_k
from reranking.reranker import rerank_documents
from hallucination.relevance_check import check_hallucinations

from typing import List, Dict

# 🔐 Set your API key


def query_pipeline(
    query: str,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 5,
    rerank_method: str = "mmr",
    hallucination_threshold: float = 0.6
) -> List[Dict]:
    """
    Full query-to-documents pipeline:
    1. Retrieve documents
    2. Rerank with MMR or cosine
    3. Run hallucination detection

    :param query: User query
    :return: List of dictionaries with content, similarity, and hallucination flag
    """
    print("🔍 Retrieving top documents...")
    retrieved_docs = retrieve_top_k(query, k=retrieval_top_k)

    print("🏗️  Reranking documents...")
    reranked_docs = rerank_documents(
        docs=retrieved_docs,
        query=query,
        method=rerank_method,
        top_k=rerank_top_k
    )

    print("🧠 Checking for hallucinations...")
    checked_results = check_hallucinations(
        query=query,
        docs=reranked_docs,
        threshold=hallucination_threshold
    )

    results = []
    for i, (doc, sim, is_hallucinated) in enumerate(checked_results, 1):
        results.append({
            "rank": i,
            "similarity": round(sim, 3),
            "hallucination": is_hallucinated,
            "content": doc.page_content[:500] + "..."
        })

    return results

# 🧪 Example usage
if __name__ == "__main__":
    query = "What is the role of floorplanning in Vivado?"
    final_results = query_pipeline(query)

    for result in final_results:
        status = "❌ Hallucination" if result["hallucination"] else "✅ Relevant"
        print(f"\n🔹 Rank {result['rank']} — Similarity: {result['similarity']} — {status}")
        print(result["content"])




