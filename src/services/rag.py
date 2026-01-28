from typing import List, Dict, Any, Optional
from uuid import UUID
from ..core.interfaces import VectorStoreRepository, DocumentLoader
from ..core.models import UploadedFile
from ..infra.ingestion import DocumentLoaderFactory

class RAGService:
    def __init__(self, vector_store: VectorStoreRepository):
        self.vector_store = vector_store

    def ingest_file(self, file_path: str) -> UploadedFile:
        """
        Loads a file, processes it, and adds it to the vector store.
        Returns the UploadedFile object.
        """
        loader = DocumentLoaderFactory.get_loader(file_path)
        document = loader.load(file_path)
        
        # TODO: Implement proper text splitting here.
        # For now, we are indexing the whole document content as one or basic splitting.
        # Ideally we would use LangChain's RecursiveCharacterTextSplitter
        
        # Simple splitting mock for PoC
        self.vector_store.add_documents([document])
        return document

    def retrieve_context(self, query: str, top_k: int = 5, file_ids: Optional[List[UUID]] = None) -> str:
        """
        Retrieves relevant context from the vector store.
        Formats the results into a single context string.
        """
        results = self.vector_store.query(query, top_k=top_k, filter_file_ids=file_ids)
        
        context_parts = []
        for res in results:
            content = res.get('content', '')
            metadata = res.get('metadata', {})
            filename = metadata.get('filename', 'Unknown File')
            context_parts.append(f"--- SOURCE: {filename} ---\n{content}\n")
            
        return "\n".join(context_parts)
