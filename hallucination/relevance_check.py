# Check answer against context for hallucination
# rag_project/evaluation/hallucination_checker.py

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔐 Set your OpenAI API Key
openai_api_key=""

def check_hallucinations(
    query: str,
    docs: List[Document],
    threshold: float = 0.5
) -> List[Tuple[Document, float, bool]]:
    """
    Compare query against each document's content using cosine similarity.

    Returns a list of (Document, similarity_score, is_hallucination)
    """
    embeddings = OpenAIEmbeddings(openai_api_key="")

    query_emb = np.array(embeddings.embed_query(query))
    doc_embs = embeddings.embed_documents([doc.page_content for doc in docs])
    doc_embs = np.array(doc_embs)

    sims = cosine_similarity([query_emb], doc_embs)[0]
    print(f"value of sim {sims}")

    results = []
    for doc, sim in zip(docs, sims):
        is_hallucinated = sim < threshold
        results.append((doc, sim, is_hallucinated))
    # print(f"value of results: {results[0]}")
    print(results[0][0].id)
    print(results[0][1])
    print(results[0][2])
    # print(results[0][3])
    return results


# 🧪 Example usage
if __name__ == "__main__":
    from retrieval.retrieve_topk import retrieve_top_k

    query = input("Enter user Query:::")
    top_docs = retrieve_top_k(query)
    print(type(top_docs))

    results = check_hallucinations(query, top_docs, threshold=0.6)

    for i, (doc, score, is_fake) in enumerate(results, 1):
        status = "❌ Hallucination" if is_fake else "✅ Relevant"
        print(f"\n🔹 Document {i} — Similarity: {score:.2f} — {status}")
        print(doc.page_content[:300], "...")



