class VectorStoreNotFoundError(Exception):
    """FAISS vector database is missing."""
    pass

class NoDocumentsFoundError(Exception):
    """Raised when no supported documents are found for ingestion."""
    pass

class BM25IndexNotFoundError(Exception):
    """BM25 index is missing."""
    pass

class PromptTemplateNotFoundError(Exception):
    """Prompt template file is missing."""
    pass

class EmbeddingModelNotFoundError(Exception):
    """Embedding model could not be loaded."""
    pass

class DocumentNotFoundError(Exception):
    """Requested document does not exist."""
    pass

class InvalidDocumentFormatError(Exception):
    """Uploaded document format is not supported."""
    pass

class IngestionFailedError(Exception):
    """Document ingestion failed."""
    pass

class RerankingFailedError(Exception):
    """Cross-encoder reranking failed."""
    pass

class AnswerGenerationError(Exception):
    """LLM failed to generate an answer."""
    pass

class GroundingCheckFailedError(Exception):
    """Grounding verification failed."""
    pass