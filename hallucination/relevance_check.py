import os
from dotenv import load_dotenv
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

def check_hallucinations(
    query: str,
    docs: List[Document],
    threshold: float = 0.5
) -> List[Tuple[Document, float, bool]]:

    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    query_emb = np.array(embeddings.embed_query(query))
    doc_embs = np.array(
        embeddings.embed_documents([doc.page_content for doc in docs])
    )

    sims = cosine_similarity([query_emb], doc_embs)[0]

    results = []
    for doc, sim in zip(docs, sims):
        is_hallucinated = sim < threshold
        results.append((doc, sim, is_hallucinated))

    return results


if __name__ == "__main__":
    from retrieval.retrieve_topk import retrieve_top_k

    query = input("Enter user Query::: ")
    top_docs = retrieve_top_k(query)

    results = check_hallucinations(query, top_docs, threshold=0.6)

    for i, (doc, score, is_fake) in enumerate(results, 1):
        status = "❌ Hallucination" if is_fake else "✅ Relevant"
        print(f"\n🔹 Document {i} — Similarity: {score:.2f} — {status}")
        print(doc.page_content[:300], "...")