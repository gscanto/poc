# main.py - Backend FastAPI para AutoReportAI
import os
import json
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import markdown2

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da aplicação
app = FastAPI(title="AutoReportAI", version="1.0.0")

# CORS para permitir acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class ReportRequest(BaseModel):
    title: str
    context: str
    sections: List[str]
    style: str = "technical"
    reference_format: str = "IEEE"
    retrieve_references: bool = True
    top_k: int = 6

class ReportResponse(BaseModel):
    report_id: str
    content: str
    references: List[dict]
    generation_time: float
    tokens_used: int

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = None
        self.llm_model = None
        self.tokenizer = None
        self.faiss_index = None
        self.corpus_metadata = []
        self.model_name = None
        
        logger.info(f"Device: {self.device}")
        if self.device == "cuda":
            try:
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            except Exception:
                pass

    # ========= EMBEDDING =========
    def load_embedding_model(self, model_name="all-MiniLM-L6-v2"):
        """Carrega modelo de embeddings"""
        if self.embedding_model is not None:
            return
        logger.info(f"Carregando modelo de embeddings: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        if self.device == "cuda":
            try:
                self.embedding_model = self.embedding_model.to(self.device)
            except Exception:
                logger.warning("Não foi possível mover embedding para CUDA; executando em CPU.")
        logger.info("Modelo de embeddings carregado com sucesso")


    def load_llm(self, model_name="google/gemma-2b-it"):#"tiiuae/falcon-7b-instruct"):#"meta-llama/Llama-2-7b-chat-hf"):#
        """Carrega LLM de forma compatível, mesmo sem bitsandbytes"""
        logger.info(f"Carregando LLM: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            logger.info("LLM carregado com sucesso (sem quantização 4-bit)")
        except Exception as e:
            logger.error(f"Erro ao carregar LLM principal: {e}")
            logger.info("Tentando modelo alternativo menor...")
            fallback = "microsoft/phi-2"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                fallback,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            logger.info("Modelo alternativo carregado com sucesso.")

    # # ========= LLM =========
    # def load_llm(self, model_name="google/gemma-2b-it"):
    #     """Carrega modelo de linguagem com fallback automático"""
    #     if self.llm_model is not None and self.tokenizer is not None:
    #         return

    #     logger.info(f"Carregando LLM: {model_name}")
    #     self.model_name = model_name

    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=True,
    #     )

    #     try:
    #         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #         self.llm_model = AutoModelForCausalLM.from_pretrained(
    #             model_name,
    #             quantization_config=bnb_config,
    #             device_map="auto",
    #             trust_remote_code=True,
    #         )
    #         logger.info(f"LLM '{model_name}' carregado com sucesso.")
    #     except Exception as e:
    #         logger.error(f"Erro ao carregar LLM principal ({model_name}): {e}")
    #         fallback = "microsoft/phi-2"
    #         try:
    #             logger.info(f"Tentando modelo alternativo: {fallback}")
    #             self.tokenizer = AutoTokenizer.from_pretrained(fallback)
    #             self.llm_model = AutoModelForCausalLM.from_pretrained(
    #                 fallback,
    #                 quantization_config=bnb_config,
    #                 device_map="auto",
    #                 trust_remote_code=True,
    #             )
    #             self.model_name = fallback
    #             logger.info("Modelo alternativo carregado com sucesso.")
    #         except Exception as e2:
    #             logger.critical(f"Falha ao carregar fallback ({fallback}): {e2}")
    #             self.llm_model = None
    #             self.tokenizer = None

    # ========= FAISS =========
    def create_faiss_index(self, dimension=384):
        """Cria índice FAISS"""
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(dimension)
            logger.info("Índice FAISS criado")

    def add_documents_to_index(self, documents: List[dict]):
        """Adiciona documentos ao índice FAISS com proteção contra None"""
        if self.embedding_model is None:
            self.load_embedding_model()

        # Garantir que o texto não seja None
        texts = [doc.get('text', "") or "" for doc in documents]

        # Gerar embeddings
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Verificar se embeddings retornou algo
        if embeddings is None or len(embeddings) == 0:
            logger.warning("Nenhum embedding foi gerado. Nenhum documento adicionado.")
            return

        # Normalizar e adicionar ao FAISS
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        self.corpus_metadata.extend(documents)
        logger.info(f"{len(documents)} documentos adicionados (total: {self.faiss_index.ntotal})")


    def retrieve_documents(self, query: str, top_k: int = 6):
        """Recupera documentos relevantes com proteção"""
        if (self.faiss_index is None) or (self.faiss_index.ntotal == 0):
            return []

        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        if query_embedding is None or len(query_embedding) == 0:
            logger.warning("Embedding da query retornou None")
            return []

        faiss.normalize_L2(query_embedding)
        k = min(top_k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.corpus_metadata):
                doc = self.corpus_metadata[idx].copy()
                doc["score"] = float(score)
                results.append(doc)
        return results


    # ========= GERAÇÃO =========
    def generate_section(self, section_name: str, context: str, retrieved_docs: List[dict], style: str):
        """Gera conteúdo da seção com fallback e prompt melhorado"""
        if self.llm_model is None or self.tokenizer is None:
            raise RuntimeError("Modelo de linguagem não está carregado. Chame load_llm() antes de gerar.")

        try:
            # Configurar pad_token se não existir
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Limitar contexto para evitar overflow
            context = context[:4000]

            # Construir contexto resumido das referências
            references_context = "\n\n".join([
                f"[Source {i+1}] {doc.get('title', 'Documento')}: {doc['text'][:400]}..."
                for i, doc in enumerate(retrieved_docs[:5])
            ])

            prompt = self._build_prompt(section_name, context, references_context, style)

            # Tokenizar com atenção_mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096,
                padding=True,
                return_attention_mask=True
            )
            
            # Mover tensores para o device correto
            inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}

            # Garantir que input_ids existe
            if 'input_ids' not in inputs or inputs['input_ids'] is None:
                raise ValueError("Tokenização falhou: input_ids é None")

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=512,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Verificar se outputs é válido
            if outputs is None or len(outputs) == 0:
                raise ValueError("Modelo retornou outputs vazio")

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remover qualquer resto de [INST]
            text = text.replace("<s>", "").replace("</s>", "").strip()
            if "[/INST]" in text:
                text = text.split("[/INST]")[-1].strip()
            
            # Garantir que temos um texto válido
            if not text or len(text.strip()) == 0:
                text = f"Seção '{section_name}': {context[:300]}..."
            
            return text, len(outputs[0])

        except torch.cuda.OutOfMemoryError:
            logger.warning("⚠️ Memória insuficiente. Reduzindo prompt e tentando novamente...")
            torch.cuda.empty_cache()
            short_context = context[:1500]
            short_prompt = self._build_prompt(section_name, short_context, "", style)
            inputs = self.tokenizer(short_prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=256,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text.strip(), len(outputs[0])
        except Exception as e:
            logger.error(f"Erro ao gerar seção '{section_name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (
                f"⚠️ Esta seção '{section_name}' foi gerada em modo reduzido. Resumo: {context[:200]}...",
                0,
            )


    def _build_prompt(self, section_name: str, context: str, references: str, style: str):
        """Constrói prompt claro para o LLM evitando repetições"""
        style_instructions = {
            "technical": "Use linguagem técnica precisa e formal.",
            "concise": "Seja direto e objetivo.",
            "detailed": "Forneça explicações detalhadas e bem estruturadas."
        }
        style_text = style_instructions.get(style, "Use linguagem técnica precisa e formal.")

        prompt = f"""[INST] <<SYS>>
    You are an expert assistant specialized in writing high-quality academic technical reports.
    <</SYS>>

    Task: Write the section "{section_name}" of a technical report.

    Context:
    {context}

    Reference documents:
    {references}

    Instructions:
    - {style_text}
    - Include relevant technical details and organize the content in 2–4 clear paragraphs.
    - Cite sources using [Source X] when appropriate.
    - Maintain a professional, precise, and objective tone.
    - Do NOT fabricate information that is not present in the context or references.

    Write now the section "{section_name}":
    [/INST]
    """
        return prompt


# Instância global do gerenciador
model_manager = ModelManager()

# Rotas da API
@app.on_event("startup")
async def startup_event():
    """Inicialização dos modelos"""
    logger.info("Inicializando AutoReportAI...")
    
    try:
        model_manager.load_embedding_model()
        model_manager.create_faiss_index()
        
        # Carregar corpus inicial de exemplo
        sample_corpus = load_sample_corpus()
        if sample_corpus:
            model_manager.add_documents_to_index(sample_corpus)
        
        # Carregar LLM quantizado
        model_manager.load_llm()
        
        logger.info("AutoReportAI inicializado com sucesso!")
    except Exception as e:
        logger.error(f"Erro na inicialização: {e}")


@app.get("/")
async def root():
    return {
        "name": "AutoReportAI",
        "version": "1.0.0",
        "status": "running",
        "device": model_manager.device,
        "documents_indexed": model_manager.faiss_index.ntotal if model_manager.faiss_index else 0
    }

# @app.post("/generate-report", response_model=ReportResponse)
# async def generate_report(request: ReportRequest):
#     """Gera relatório técnico completo"""
#     start_time = datetime.now()
#     logger.info(f"Gerando relatório: {request.title}")
    
#     try:
#         report_sections = []
#         all_references = {}
#         total_tokens = 0
        
#         # Gerar cada seção
#         for section in request.sections:
#             logger.info(f"Gerando seção: {section}")
            
#             # Recuperar documentos relevantes
#             query = f"{request.context} {section}"
#             retrieved_docs = model_manager.retrieve_documents(query, request.top_k)
            
#             # Adicionar referências únicas
#             for doc in retrieved_docs:
#                 ref_id = doc.get('id', f"ref_{len(all_references)}")
#                 if ref_id not in all_references:
#                     all_references[ref_id] = doc
            
#             # Gerar conteúdo da seção
#             if model_manager.llm_model:
#                 section_content, tokens = model_manager.generate_section(
#                     section, request.context, retrieved_docs, request.style
#                 )
#                 total_tokens += tokens
#             else:
#                 # Fallback se LLM não estiver carregado
#                 section_content = generate_fallback_section(section, request.context, retrieved_docs)
#                 total_tokens += 200
            
#             report_sections.append({
#                 "title": section,
#                 "content": section_content
#             })
        
#         # Montar relatório completo
#         report_content = format_report(
#             request.title,
#             report_sections,
#             list(all_references.values()),
#             request.reference_format
#         )
        
#         generation_time = (datetime.now() - start_time).total_seconds()
#         report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
#         # Salvar relatório
#         save_report(report_id, report_content, list(all_references.values()))
        
#         logger.info(f"Relatório gerado em {generation_time:.2f}s")
        
#         return ReportResponse(
#             report_id=report_id,
#             content=report_content,
#             references=list(all_references.values()),
#             generation_time=generation_time,
#             tokens_used=total_tokens
#         )
    
#     except Exception as e:
#         logger.error(f"Erro ao gerar relatório: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# ========== ENDPOINT CORRIGIDO ==========
@app.post("/generate-report", response_model=ReportResponse)
def generate_report(request: ReportRequest):
    start_time = datetime.now()

    try:
        # Garante que o modelo está carregado
        if model_manager.llm_model is None:
            logger.warning("Modelo não estava carregado. Carregando agora...")
            model_manager.load_llm()

        # Recupera documentos e gera seções
        retrieved_docs = model_manager.retrieve_documents(request.context, top_k=request.top_k)
        sections = []
        total_tokens = 0

        for sec in request.sections:
            try:
                content, tokens = model_manager.generate_section(sec, request.context, retrieved_docs, request.style)
            except Exception as e:
                logger.warning(f"Fallback usado para '{sec}': {e}")
                content = generate_fallback_section(sec, request.context, retrieved_docs)
                tokens = 0
            sections.append({"title": sec, "content": content})
            total_tokens += tokens

        # Montar e salvar relatório
        content_md = format_report(request.title, sections, retrieved_docs, request.reference_format)
        report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_report(report_id, content_md, retrieved_docs)

        return ReportResponse(
            report_id=report_id,
            content=content_md,
            references=retrieved_docs,
            generation_time=(datetime.now() - start_time).total_seconds(),
            tokens_used=total_tokens,
        )

    except Exception as e:
        logger.exception("Erro inesperado ao gerar relatório.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-documents")
async def upload_documents(documents: List[dict]):
    """Adiciona documentos ao corpus"""
    try:
        model_manager.add_documents_to_index(documents)
        return {
            "status": "success",
            "documents_added": len(documents),
            "total_documents": model_manager.faiss_index.ntotal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/corpus-stats")
async def corpus_stats():
    """Estatísticas do corpus"""
    return {
        "total_documents": model_manager.faiss_index.ntotal if model_manager.faiss_index else 0,
        "embedding_dimension": 384,
        "device": model_manager.device
    }

# Funções auxiliares
def load_sample_corpus():
    """Carrega corpus de exemplo expandido com temas de IA, ML, DL e MLOps"""
    return [
        {
            "id": "doc1",
            "title": "Machine Learning Fundamentals",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming. Key concepts include supervised learning, unsupervised learning, and reinforcement learning.",
            "source": "IEEE Transactions",
            "year": 2024
        },
        {
            "id": "doc2",
            "title": "Deep Neural Networks",
            "text": "Deep neural networks consist of multiple layers that progressively extract higher-level features from raw input. The architecture enables learning complex patterns through backpropagation and gradient descent optimization.",
            "source": "arXiv",
            "year": 2024
        },
        {
            "id": "doc3",
            "title": "RAG Systems Architecture",
            "text": "Retrieval-Augmented Generation combines information retrieval with language generation. The system retrieves relevant documents from a knowledge base and uses them to ground the generation process, reducing hallucinations.",
            "source": "ACM Computing Surveys",
            "year": 2023
        },
        {
            "id": "doc4",
            "title": "Time Series Forecasting Techniques",
            "text": "Time series forecasting methods include ARIMA, Prophet, and LSTM-based models. They capture temporal dependencies and seasonality to predict future values in industrial, financial, and energy domains.",
            "source": "Journal of Data Science",
            "year": 2022
        },
        {
            "id": "doc5",
            "title": "MLOps Best Practices",
            "text": "MLOps integrates machine learning workflows into software engineering pipelines. It emphasizes model versioning, reproducibility, continuous integration, and monitoring to ensure production reliability.",
            "source": "Google Cloud Whitepaper",
            "year": 2023
        },
        {
            "id": "doc6",
            "title": "Computer Vision Applications",
            "text": "Computer vision models interpret visual information to perform object detection, classification, and segmentation. Convolutional Neural Networks (CNNs) are widely used in autonomous vehicles and medical imaging.",
            "source": "IEEE CVPR",
            "year": 2023
        },
        {
            "id": "doc7",
            "title": "Natural Language Processing Advances",
            "text": "Recent advances in NLP leverage transformer architectures like BERT, GPT, and T5. These models enable contextual understanding and generation across diverse language tasks.",
            "source": "ACL Anthology",
            "year": 2024
        },
        {
            "id": "doc8",
            "title": "Explainable AI",
            "text": "Explainable AI techniques provide transparency by interpreting model decisions. Methods such as SHAP, LIME, and attention visualization help stakeholders understand predictions and reduce bias.",
            "source": "Nature Machine Intelligence",
            "year": 2023
        },
        {
            "id": "doc9",
            "title": "Data Quality in AI Systems",
            "text": "Data quality directly affects model performance. Key practices include outlier detection, feature normalization, and bias mitigation to ensure reliable and fair AI outcomes.",
            "source": "IBM Research Journal",
            "year": 2022
        },
        {
            "id": "doc10",
            "title": "Edge AI Deployment",
            "text": "Edge AI deploys machine learning models on local devices for real-time inference. It reduces latency and enhances privacy, particularly in IoT and embedded systems.",
            "source": "IEEE Internet of Things Journal",
            "year": 2023
        },
        {
            "id": "doc11",
            "title": "Reinforcement Learning in Robotics",
            "text": "Reinforcement learning enables robots to learn optimal control strategies through trial and error. Policy gradients and Q-learning are fundamental methods in autonomous decision-making.",
            "source": "Robotics and Automation Letters",
            "year": 2023
        },
        {
            "id": "doc12",
            "title": "Ethics and Governance in AI",
            "text": "Ethical AI frameworks promote fairness, accountability, and transparency. Governance practices are critical to mitigate risks related to privacy, discrimination, and misuse of automated systems.",
            "source": "OECD AI Policy Observatory",
            "year": 2024
        }
    ]


def generate_fallback_section(section_name: str, context: str, docs: List[dict]) -> str:
    """Gera seção sem LLM (para demonstração)"""
    content = f"Esta seção de {section_name} aborda aspectos importantes relacionados ao contexto apresentado. "
    
    if docs:
        content += "Com base nos documentos recuperados, observamos que "
        content += " ".join([doc['text'][:100] for doc in docs[:2]])
        content += "... "
    
    content += f"A análise considera {context[:200]}... "
    content += "Os resultados demonstram aplicabilidade prática dos conceitos apresentados."
    
    return content

def _format_single_reference(ref: dict, idx: int, ref_format: str = "IEEE") -> str:
    """Formata uma referência simples a partir do metadado do documento"""
    title = ref.get('title', 'Untitled')
    source = ref.get('source', ref.get('source', 'Unknown Source'))
    year = ref.get('year', '')
    authors = ref.get('authors', None)
    
    if ref_format.upper() == "IEEE":
        # IEEE: [idx] Author(s), "Title", Source, year.
        author_str = ""
        if authors:
            if isinstance(authors, list):
                author_str = ", ".join(authors)
            else:
                author_str = str(authors)
            author_str += ", "
        return f"[{idx}] {author_str}\"{title}\", {source}, {year}."
    else:
        # Default simple APA-like
        author_str = ""
        if authors:
            if isinstance(authors, list):
                author_str = ", ".join(authors)
            else:
                author_str = str(authors)
            author_str += ". "
        return f"{author_str}{title}. {source}, {year}."

def format_report(title: str, sections: List[dict], references: List[dict], ref_format: str) -> str:
    """Formata relatório em Markdown"""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = []
    report_lines.append(f"# {title}")
    report_lines.append("")
    report_lines.append(f"**Data:** {date_str}")
    report_lines.append("")
    
    # Sumário (toc simples)
    report_lines.append("## Sumário")
    for i, sec in enumerate(sections, start=1):
        report_lines.append(f"{i}. {sec['title']}")
    report_lines.append("")
    
    # Seções
    for sec in sections:
        report_lines.append(f"## {sec['title']}")
        report_lines.append("")
        report_lines.append(sec['content'])
        report_lines.append("")
    
    # Referências
    report_lines.append("## Referências")
    report_lines.append("")
    if references:
        for idx, ref in enumerate(references, start=1):
            formatted = _format_single_reference(ref, idx, ref_format)
            report_lines.append(formatted)
    else:
        report_lines.append("Nenhuma referência recuperada.")
    
    return "\n".join(report_lines)

def save_report(report_id: str, content_md: str, references: Optional[List[dict]] = None):
    """Salva o relatório em disco (Markdown e DOCX)"""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    md_path = reports_dir / f"{report_id}.md"
    docx_path = reports_dir / f"{report_id}.docx"
    
    # Salvar Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content_md)
    
    # Converter para DOCX básico
    try:
        doc = Document()
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)
        
        lines = content_md.splitlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith("# "):
                hdr = doc.add_heading(level=1)
                hdr_run = hdr.add_run(line[2:].strip())
                hdr_run.font.size = Pt(16)
            elif line.startswith("## "):
                hdr = doc.add_heading(level=2)
                hdr_run = hdr.add_run(line[3:].strip())
                hdr_run.font.size = Pt(14)
            elif line.strip() == "":
                doc.add_paragraph("")  # blank line
            else:
                p = doc.add_paragraph(line)
                p_format = p.paragraph_format
                p_format.space_after = Pt(6)
        
        # Se desejar, adiciona anexo de referências estruturadas no final (já estão no markdown)
        doc.save(docx_path)
        logger.info(f"Relatório salvo: {md_path} e {docx_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar DOCX: {e}")

