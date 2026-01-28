from typing import List, Optional, Dict, Any
from uuid import uuid4, UUID
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"

class UploadedFile(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    file_type: FileType
    content: str
    upload_date: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class DocumentSection(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str = ""
    order: int
    specific_context_file_ids: List[UUID] = Field(default_factory=list, description="IDs of files explicitly linked to this section")
    generated_content: Optional[str] = None
    
class GenerationContext(BaseModel):
    global_context: str
    tone: str = "Technical"
    target_audience: str = "General"

class Project(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    sections: List[DocumentSection] = Field(default_factory=list)
    files: List[UploadedFile] = Field(default_factory=list)
    context: GenerationContext
    created_at: datetime = Field(default_factory=datetime.now)
