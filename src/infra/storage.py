import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from uuid import UUID
from ..core.interfaces import VectorStoreRepository
from ..core.models import UploadedFile
import logging

logger = logging.getLogger(__name__)

class ChromaDBRepository(VectorStoreRepository):
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, documents: List[UploadedFile]) -> None:
        """
        Add documents to the ChromaDB collection. 
        Note: Real-world usage would require chunking before adding.
        Here we assume `content` is already a chunk or we index the whole file (not ideal for RAG).
        For this PoC, we will split by simple paragraphs in the Service layer, 
        but if we receive them here, we index them.
        """
        if not documents:
            return

        ids = [str(doc.id) for doc in documents]
        documents_text = [doc.content for doc in documents]
        metadatas = [{"filename": doc.filename, "file_id": str(doc.id), "type": doc.file_type.value} for doc in documents]

        # ChromaDB requires non-empty lists
        if ids:
            self.collection.add(
                documents=documents_text,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB.")

    def query(self, query_text: str, top_k: int = 5, filter_file_ids: Optional[List[UUID]] = None) -> List[Dict[str, Any]]:
        where_clause = None
        if filter_file_ids:
            # Construct a where clause to filter by file_id
            # ChromaDB 'where' supports $in operator
            where_clause = {"file_id": {"$in": [str(fid) for fid in filter_file_ids]}}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_clause
        )
        
        # Flatten results
        flattened_results = []
        if results and results['documents']:
            num_results = len(results['documents'][0])
            for i in range(num_results):
                flattened_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "id": results['ids'][0][i]
                })
        
        return flattened_results
