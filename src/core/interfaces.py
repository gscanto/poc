from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from uuid import UUID
from .models import UploadedFile, DocumentSection

class LLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text based on a prompt."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the underlying model."""
        pass

class VectorStoreRepository(ABC):
    @abstractmethod
    def add_documents(self, documents: List[UploadedFile]) -> None:
        """Index a list of documents."""
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5, filter_file_ids: Optional[List[UUID]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks.
        
        Args:
            query_text: The query string.
            top_k: Number of results to return.
            filter_file_ids: Optional list of file IDs to restrict search scope.
        """
        pass

class DocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> UploadedFile:
        """Load and parse a file from disk."""
        pass
    
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions (e.g. ['.pdf'])."""
        pass
