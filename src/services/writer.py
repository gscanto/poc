from typing import List, Optional
from uuid import UUID
from ..core.models import Project, DocumentSection, GenerationContext
from ..core.interfaces import LLMProvider
from .rag import RAGService
import logging

logger = logging.getLogger(__name__)

class DocumentWriterService:
    def __init__(self, llm_provider: LLMProvider, rag_service: RAGService):
        self.llm = llm_provider
        self.rag = rag_service

    def generate_section_content(self, section: DocumentSection, project: Project) -> str:
        """
        Generates content for a specific section using RAG and LLM.
        """
        # 1. Retrieve Context
        # Query uses section title + description + global context
        query = f"{section.title}. {section.description}. {project.context.global_context}"
        
        # Prioritize specific files if linked
        retrieved_text = self.rag.retrieve_context(
            query, 
            top_k=5, 
            file_ids=section.specific_context_file_ids if section.specific_context_file_ids else None
        )
        
        # Fallback: if no specific files or strictly needed, maybe query global? 
        # Current logic: If specific files are set, ONLY uses them (via RAGService filter). 
        # If we want a fallback to global, we'd need to checking if results are empty.
        
        if not retrieved_text and section.specific_context_file_ids:
             # Try querying global if specific failed or returned nothing (optional strategy)
             pass

        # 2. Construct Prompt
        prompt = self._build_prompt(section, project, retrieved_text)
        
        # 3. Generate
        logger.info(f"Generating content for section: {section.title}")
        content = self.llm.generate_text(prompt)
        
        section.generated_content = content
        return content

    def _build_prompt(self, section: DocumentSection, project: Project, context_text: str) -> str:
        return f"""
You are an expert technical writer.
Target Audience: {project.context.target_audience}
Tone: {project.context.tone}

Global Document Context:
{project.context.global_context}

current Section to write: "{section.title}"
Section Description/Goal: {section.description}

Reference Material (Content retrieved from user documents):
{context_text}

Instructions:
- Write the content for ONLY this section.
- Use the reference material to support your writing.
- Maintain consistency with the global context.
- Do NOT output the section title again, just the content.
- Cite sources if applicable (e.g. [Source: filename]).

Write the content now:
"""
