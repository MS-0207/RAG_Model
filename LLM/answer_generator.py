from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI


def build_context(docs: List[Document]) -> str:
    return "\n\n".join(
        [
            f"Document {i+1}\n"
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Content:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ]
    )


def generate_answer(
    query: str,
    top_docs: List[Document],
    model_name: str = "gpt-4o-mini"
) -> Dict[str, Any]:

    llm = ChatOpenAI(model=model_name, temperature=0)
    parser = JsonOutputParser()
    context = build_context(top_docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict RAG assistant. Answer only using the provided context. "
            "If the context does not contain the answer, say you could not find it in the provided context."
        ),
        (
            "human",
            """
User Query:
{query}

Retrieved Context:
{context}

Return JSON with exactly these keys:
- query
- answer
- sources

{format_instructions}
"""
        )
    ])

    chain = prompt | llm | parser

    return chain.invoke({
        "query": query,
        "context": context,
        "format_instructions": parser.get_format_instructions()
    })

def check_grounding(
    query: str,
    answer: str,
    top_docs: List[Document],
    model_name: str = "gpt-4o-mini"
) -> Dict[str, Any]:

    llm = ChatOpenAI(model=model_name, temperature=0)
    parser = JsonOutputParser()

    context = build_context(top_docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grounding checker for a RAG system. "
            "Check whether the answer is fully supported by the retrieved context."
        ),
        (
            "human",
            """
User Query:
{query}

Generated Answer:
{answer}

Retrieved Context:
{context}

Return JSON with exactly these keys:
- verdict
- explanation
- unsupported_claims

verdict must be either SUPPORTED or HALLUCINATED.

{format_instructions}
"""
        )
    ])

    chain = prompt | llm | parser

    return chain.invoke({
        "query": query,
        "answer": answer,
        "context": context,
        "format_instructions": parser.get_format_instructions()
    })