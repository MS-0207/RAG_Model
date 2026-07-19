from typing import List, Dict, Any
from langchain_core.documents import Document


def build_final_response(
    query: str,
    answer_result: Dict[str, Any],
    grounding_result: Dict[str, Any],
    top_docs: List[Document],
) -> Dict[str, Any]:
    sources = []
    for i, doc in enumerate(top_docs, start=1):
        sources.append(
            {
                "source_id": i,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
                "preview": doc.page_content[:250],
            }
        )

    verdict = grounding_result.get("verdict", "UNKNOWN")

    if verdict == "SUPPORTED":
        confidence = "HIGH"
    elif verdict == "HALLUCINATED":
        confidence = "LOW"
    else:
        confidence = "MEDIUM"

    final_response = {
        "query": query,
        "answer": answer_result.get("answer", ""),
        "sources": sources,
        "confidence": confidence,
        "grounding_verdict": verdict,
        "grounding_explanation": grounding_result.get("explanation", ""),
        "unsupported_claims": grounding_result.get("unsupported_claims", []),
    }

    return final_response
