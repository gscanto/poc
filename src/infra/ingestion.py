import fitz  # PyMuPDF
import docx
from pathlib import Path
from typing import List, Dict, Type
from ..core.interfaces import DocumentLoader
from ..core.models import UploadedFile, FileType

class PDFLoader(DocumentLoader):
    def load(self, file_path: str) -> UploadedFile:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        return UploadedFile(
            filename=Path(file_path).name,
            file_type=FileType.PDF,
            content=text
        )

    def supported_extensions(self) -> List[str]:
        return [".pdf"]

class TextLoader(DocumentLoader):
    def load(self, file_path: str) -> UploadedFile:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        return UploadedFile(
            filename=Path(file_path).name,
            file_type=FileType.TXT,
            content=text
        )

    def supported_extensions(self) -> List[str]:
        return [".txt", ".md"]

class DocxLoader(DocumentLoader):
    def load(self, file_path: str) -> UploadedFile:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        
        return UploadedFile(
            filename=Path(file_path).name,
            file_type=FileType.DOCX,
            content=text
        )

    def supported_extensions(self) -> List[str]:
        return [".docx"]

class DocumentLoaderFactory:
    _loaders: Dict[str, Type[DocumentLoader]] = {}
    
    @classmethod
    def register_loader(cls, loader_cls: Type[DocumentLoader]):
        loader_instance = loader_cls()
        for ext in loader_instance.supported_extensions():
            cls._loaders[ext] = loader_cls

    @classmethod
    def get_loader(cls, file_path: str) -> DocumentLoader:
        ext = Path(file_path).suffix.lower()
        loader_cls = cls._loaders.get(ext)
        if not loader_cls:
            raise ValueError(f"No loader found for extension: {ext}")
        return loader_cls()

# Register default loaders
DocumentLoaderFactory.register_loader(PDFLoader)
DocumentLoaderFactory.register_loader(TextLoader)
DocumentLoaderFactory.register_loader(DocxLoader)
