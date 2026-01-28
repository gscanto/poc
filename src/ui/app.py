import streamlit as st
import sys
from pathlib import Path
from uuid import uuid4

# Add project root to path so we can import src modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.core.models import Project, DocumentSection, GenerationContext, FileType
from src.infra.llm import OllamaClient
from src.infra.storage import ChromaDBRepository
from src.services.rag import RAGService
from src.services.writer import DocumentWriterService

# Initialize Services
# In a real app, use st.cache_resource for singletons
@st.cache_resource
def get_services():
    llm = OllamaClient(model_name="llama3") # User can change this in settings
    # Initialize DB (in-memory or persistent)
    storage = ChromaDBRepository(persist_directory=str(project_root / "chroma_db"))
    rag_service = RAGService(storage)
    writer_service = DocumentWriterService(llm, rag_service)
    return llm, rag_service, writer_service

llm_client, rag_service, writer_service = get_services()

st.set_page_config(page_title="RAG Tech Writer", layout="wide", page_icon="📝")

# --- Session State Management ---
if "project" not in st.session_state:
    st.session_state.project = Project(
        title="Untitled Document",
        context=GenerationContext(global_context="")
    )

if "files_processed" not in st.session_state:
    st.session_state.files_processed = set()

# --- Sidebar: Configuration & Upload ---
with st.sidebar:
    st.title("⚙️ Settings")
    model_name = st.selectbox("LLM Model", ["llama3", "mistral", "mixtral"], index=0)
    llm_client.model_name = model_name
    
    st.divider()
    
    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx", "md"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.files_processed:
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    # Save to temp file for loader
                    temp_path = project_root / "temp_uploads"
                    temp_path.mkdir(exist_ok=True)
                    file_path = temp_path / uploaded_file.name
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Ingest
                    doc_obj = rag_service.ingest_file(str(file_path))
                    st.session_state.project.files.append(doc_obj)
                    st.session_state.files_processed.add(uploaded_file.name)
                st.success(f"Indexed {uploaded_file.name}")

    st.info(f"📚 {len(st.session_state.project.files)} documents indexed.")

# --- Main Area ---
st.title("📝 AI Technical Document Writer")

# 1. Scope & Context
with st.expander("Step 1: Define Scope & Global Context", expanded=True):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state.project.title = st.text_input("Document Title", value=st.session_state.project.title)
        st.session_state.project.context.target_audience = st.text_input("Target Audience", value="Technical Team")
    with col2:
        st.session_state.project.context.tone = st.selectbox("Tone", ["Technical", "Academic", "Business"], index=0)
    
    st.session_state.project.context.global_context = st.text_area(
        "Global Context / Abstract", 
        value=st.session_state.project.context.global_context,
        height=150,
        placeholder="Describe the overall goal of this document..."
    )

# 2. Structure Definition
st.header("Step 2: Document Structure")

# Add new section
with st.form("add_section_form"):
    c1, c2 = st.columns([3, 1])
    new_sec_title = c1.text_input("New Section Title")
    submitted = st.form_submit_button("Add Section")
    if submitted and new_sec_title:
        new_sec = DocumentSection(
            id=uuid4(),
            title=new_sec_title, 
            order=len(st.session_state.project.sections)
        )
        st.session_state.project.sections.append(new_sec)
        st.rerun()

# List sections
for idx, section in enumerate(st.session_state.project.sections):
    with st.expander(f"📌 {idx+1}. {section.title}", expanded=False):
        # Section Context
        section.description = st.text_area(f"Description / Instructions ({section.title})", value=section.description, key=f"desc_{section.id}")
        
        # Link specific files
        file_options = {f.id: f.filename for f in st.session_state.project.files}
        selected_file_ids = st.multiselect(
            "Link Specific Documents (Optional context source)",
            options=list(file_options.keys()),
            format_func=lambda x: file_options[x],
            key=f"files_{section.id}"
        )
        section.specific_context_file_ids = selected_file_ids
        
        # Generator Button for this section
        if st.button(f"Generate Content for '{section.title}'", key=f"btn_{section.id}"):
            with st.spinner("Generating..."):
                content = writer_service.generate_section_content(section, st.session_state.project)
                st.rerun()

        # Display Content
        if section.generated_content:
            st.markdown("### Generated Content")
            st.markdown(section.generated_content)
            
            # Edit?
            new_content = st.text_area("Edit Content", value=section.generated_content, height=300, key=f"edit_{section.id}")
            section.generated_content = new_content

# 3. Export
st.divider()
if st.button("Generate Full Document Preview"):
    full_doc = f"# {st.session_state.project.title}\n\n"
    for sec in st.session_state.project.sections:
        full_doc += f"## {sec.title}\n\n"
        full_doc += f"{sec.generated_content or '(Not generated yet)'}\n\n"
    
    st.markdown(full_doc)
    st.download_button("Download Markdown", full_doc, file_name=f"{st.session_state.project.title}.md")
