import streamlit as st
import requests
import json
from datetime import datetime
import time
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import pandas as pd

# URL do backend FastAPI
API_URL = "http://localhost:8000"

# Configuração da página
st.set_page_config(
    page_title="AutoReportAI - Gerador Inteligente",
    layout="wide",
    page_icon="✨",
    initial_sidebar_state="collapsed"
)

# Ícones SVG
ICONS = {
    "sparkles": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v18m9-9H3m15.364 6.364L6.636 6.636m12.728 0L6.636 17.364"/></svg>',
    "settings": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 6v6M6 12H1m6 0h6m6 0h5m-5.636-5.636l-4.243 4.243m0 0l-4.243 4.243m12.728-8.486l-4.243 4.243"/></svg>',
    "chart": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "help": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3m.08 4h.01"/></svg>',
    "upload": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4m14-7l-5-5-5 5m5-5v12"/></svg>',
    "folder": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
    "edit": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>',
    "brain": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/><path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/><path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/></svg>',
    "list": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>',
    "rocket": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>',
    "warning": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "check": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
    "download": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "refresh": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>',
    "clock": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "file": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>',
    "info": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    "database": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "save": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>',
    "trash": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
}

def icon(name, size=20):
    """Retorna um ícone SVG inline"""
    svg = ICONS.get(name, ICONS["sparkles"])
    return svg.replace('width="20"', f'width="{size}"').replace('height="20"', f'height="{size}"')

# --- CSS customizado (mantém o mesmo do original) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    body {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 50%, #ede9fe 100%);
    }
    
    .main {
        background: transparent;
    }
    
    /* Header Principal */
    .main-logo-container {
        background: linear-gradient(135deg, #312E81 0%, #4338CA 50%, #6366F1 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(49, 46, 129, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-logo-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .logo-content {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    
    .logo-icon {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .logo-text h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.02em;
    }
    
    .logo-text p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Ícones inline */
    .icon-inline {
        display: inline-block;
        vertical-align: middle;
        margin-right: 8px;
    }
    
    /* Cards e Containers */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 16px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #64748b;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
        color: #312E81;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #312E81 0%, #4338CA 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(49, 46, 129, 0.3);
    }
    
    /* Inputs */
    .stTextInput input, .stTextArea textarea {
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.875rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        background: white !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #312E81 !important;
        box-shadow: 0 0 0 3px rgba(49, 46, 129, 0.1) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #312E81 0%, #4338CA 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.875rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(49, 46, 129, 0.3) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(49, 46, 129, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Info Boxes */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.1);
    }
    
    /* Badges e Tags */
    .section-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
        color: #312E81;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #c4b5fd;
    }
    
    /* Reference Items */
    .reference-item {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border-left: 3px solid #312E81;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .reference-item:hover {
        box-shadow: 0 4px 12px rgba(49, 46, 129, 0.15);
        transform: translateX(4px);
    }
    
    /* Status Cards */
    .status-card {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }
    
    .status-card h3 {
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        opacity: 0.9;
    }
    
    .status-card p {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Metrics */
    .stMetric {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* DataFrames */
    .dataframe {
        border: none !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #312E81 0%, #4338CA 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background: #f8fafc !important;
    }
    
    .dataframe tbody tr:hover {
        background: #e0e7ff !important;
    }
</style>
""", unsafe_allow_html=True)

# Header principal com novo logo
st.markdown("""
<div class="main-logo-container">
    <div class="logo-content">
        <div class="logo-icon">
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="white" opacity="0.9"/>
                <path d="M2 17L12 22L22 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.7"/>
                <path d="M2 12L12 17L22 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.8"/>
            </svg>
        </div>
        <div class="logo-text">
            <h1>AutoReportAI v2.0</h1>
            <p>Gerador Inteligente de Documentos com Persistência Automática</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📝 Criar Documento",
    "🤖 Modelos",
    "⚙️ Sistema",
    "📊 Estatísticas",
    "💾 Persistência",
    "📚 Referências",
    "📁 Documentos"
])

# --------------------------------------------------------------------
# TAB 1 - Criar Documento
# --------------------------------------------------------------------
with tab1:
    st.markdown(f'### <span class="icon-inline">{icon("edit", 24)}</span> Informações Básicas', unsafe_allow_html=True)
    st.markdown('<div class="info-card">', unsafe_allow_html=True)

    title = st.text_input(
        "Título do documento",
        "Documento Técnico Automático",
        help="Defina um título descritivo para seu documento"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Contexto
    st.markdown(f'### <span class="icon-inline">{icon("brain", 24)}</span> Contexto do Documento', unsafe_allow_html=True)
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    context = st.text_area(
        "",
        placeholder="Descreva o tema e objetivo do seu documento...",
        height=180,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Estrutura
    st.markdown(f'### <span class="icon-inline">{icon("list", 24)}</span> Estrutura do Documento', unsafe_allow_html=True)
    st.markdown('<div class="info-card">', unsafe_allow_html=True)

    col_preset, col_custom = st.columns([1, 2])
    with col_preset:
        preset = st.radio(
            "Modelo",
            ["Personalizado", "Acadêmico", "Técnico", "Executivo"],
            label_visibility="collapsed"
        )

    with col_custom:
        if preset == "Acadêmico":
            default_sections = ["Resumo", "Introdução", "Revisão da Literatura", "Metodologia", "Resultados", "Discussão", "Conclusão", "Referências"]
        elif preset == "Técnico":
            default_sections = ["Sumário Executivo", "Introdução", "Especificações Técnicas", "Arquitetura do Sistema", "Implementação", "Testes e Validação", "Conclusão"]
        elif preset == "Executivo":
            default_sections = ["Sumário Executivo", "Contexto", "Análise", "Recomendações", "Próximos Passos"]
        else:
            default_sections = ["Introdução", "Metodologia", "Resultados", "Conclusão"]

        sections = st.text_area(
            "",
            value="\n".join(default_sections),
            height=150,
            label_visibility="collapsed"
        )

    section_list = [s.strip() for s in sections.splitlines() if s.strip()]
    if section_list:
        st.markdown("**Preview das seções:**")
        st.markdown(" ".join([f'<span class="section-badge">{s}</span>' for s in section_list]), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Botão de geração
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🚀 Gerar Documento', type="primary", use_container_width=True, key="generate_document_btn"):
            if not context.strip():
                st.markdown(f'<div class="warning-box">{icon("warning")} Por favor, preencha o contexto antes de gerar.</div>', unsafe_allow_html=True)
            elif not section_list:
                st.markdown(f'<div class="warning-box">{icon("warning")} Adicione pelo menos uma seção ao documento.</div>', unsafe_allow_html=True)
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                try:
                    payload = {"title": title, "context": context, "sections": section_list}
                    status_text.markdown(f'{icon("clock")} Gerando documento...', unsafe_allow_html=True)
                    progress_bar.progress(30)
                    
                    response = requests.post(f"{API_URL}/generate-report", json=payload, timeout=300)
                    progress_bar.progress(100)
                    
                    if response.status_code == 200:
                        data = response.json()
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.markdown(f'<div class="success-box">{icon("check")} <b>Documento gerado com sucesso!</b><br>'
                                  f'Tempo: {data["generation_time"]:.2f}s | Tokens: {data["tokens_used"]} | '
                                  f'Referências: {len(data["references"])}</div>', unsafe_allow_html=True)
                        
                        col_md, col_json = st.columns(2)
                        with col_md:
                            st.download_button(
                                label="📄 Baixar Markdown (.md)",
                                data=data["content"].encode("utf-8"),
                                file_name=f"{data['report_id']}.md",
                                mime="text/markdown",
                                use_container_width=True,
                                key="download_md_btn"
                            )
                        with col_json:
                            st.download_button(
                                label="📋 Baixar JSON Completo",
                                data=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
                                file_name=f"{data['report_id']}.json",
                                mime="application/json",
                                use_container_width=True,
                                key="download_json_btn"
                            )
                        
                        with st.expander("👀 Visualizar Documento Gerado"):
                            st.markdown(data["content"])
                    else:
                        st.error(f"❌ Erro: {response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(f"❌ Erro de conexão: {e}")
                finally:
                    progress_bar.empty()
                    status_text.empty()

# --------------------------------------------------------------------
# TAB 2 - Modelos
# --------------------------------------------------------------------
with tab2:
    st.markdown(f'### <span class="icon-inline">{icon("brain", 24)}</span> Seleção de Modelo de Linguagem', unsafe_allow_html=True)
    
    # Obter modelo atual
    try:
        model_response = requests.get(f"{API_URL}/current-model", timeout=5)
        if model_response.status_code == 200:
            model_data = model_response.json()
            current_model_name = model_data.get('model_name', 'Desconhecido')
            model_loaded = model_data.get('model_loaded', False)
            device = model_data.get('device', 'unknown')
            available_models_map = model_data.get('available_models', {})
        else:
            current_model_name = "Erro ao obter modelo"
            model_loaded = False
            device = "unknown"
            available_models_map = {}
    except:
        current_model_name = "Backend offline"
        model_loaded = False
        device = "unknown"
        available_models_map = {}
    
    # Status do modelo atual
    st.markdown("#### 🤖 Modelo Atual")
    col_model, col_device, col_status = st.columns(3)
    
    with col_model:
        status_icon = "✅" if model_loaded else "❌"
        st.markdown(f"""
        <div class='status-card' style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)'>
            <h3>Modelo</h3>
            <p style='font-size: 0.9rem;'>{status_icon}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_device:
        device_color = "#10b981" if device == "cuda" else "#3b82f6"
        st.markdown(f"""
        <div class='status-card' style='background: linear-gradient(135deg, {device_color} 0%, {device_color} 100%)'>
            <h3>Device</h3>
            <p>{device.upper()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status:
        st.markdown(f"""
        <div class='status-card' style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'>
            <h3>Status</h3>
            <p>{"🟢" if model_loaded else "🔴"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Informações do modelo atual
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(f"**Modelo carregado:** `{current_model_name}`")
    st.markdown(f"**Device de execução:** `{device}`")
    st.markdown(f"**Status:** {'✅ Carregado e pronto' if model_loaded else '❌ Não carregado (usando fallback)'}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Seleção de modelo
    st.markdown("#### 📋 Modelos Disponíveis")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    
    llm_options = [
        "Phi-2 (2.7B parâmetros)",
        "GPT-Neo-2.7B (2.7B parâmetros)",
        "GPT-J-6B (6B parâmetros)",
        "Falcon-7B-Instruct (7B parâmetros)",
        "Llama-2-7B-Chat (7B parâmetros)",
        "Mistral-7B-Instruct (7B parâmetros)"
    ]
    
    # Mapear nome amigável para nome do modelo
    model_mapping = {
        "Phi-2 (2.7B parâmetros)": "microsoft/phi-2",
        "GPT-Neo-2.7B (2.7B parâmetros)": "EleutherAI/gpt-neo-2.7B",
        "GPT-J-6B (6B parâmetros)": "EleutherAI/gpt-j-6B",
        "Falcon-7B-Instruct (7B parâmetros)": "tiiuae/falcon-7b-instruct",
        "Llama-2-7B-Chat (7B parâmetros)": "meta-llama/Llama-2-7b-chat-hf",
        "Mistral-7B-Instruct (7B parâmetros)": "mistralai/Mistral-7B-Instruct-v0.1"
    }
    
    # Encontrar seleção atual
    current_selection = "Phi-2 (2.7B parâmetros)"  # Padrão
    for friendly_name, model_id in model_mapping.items():
        if model_id == current_model_name:
            current_selection = friendly_name
            break
    
    selected_model_friendly = st.selectbox(
        "Escolha o modelo de linguagem:",
        llm_options,
        index=llm_options.index(current_selection),
        help="Modelos maiores geram texto de melhor qualidade mas requerem mais memória"
    )
    
    selected_model_id = model_mapping[selected_model_friendly]
    
    # Informações sobre o modelo selecionado
    model_info = {
        "Phi-2 (2.7B parâmetros)": {
            "size": "2.7B parâmetros",
            "memory": "~6 GB RAM/VRAM",
            "speed": "⚡⚡⚡ Rápido",
            "quality": "⭐⭐⭐ Boa",
            "description": "Modelo compacto e eficiente da Microsoft, ideal para execução local."
        },
        "GPT-Neo-2.7B (2.7B parâmetros)": {
            "size": "2.7B parâmetros",
            "memory": "~6 GB RAM/VRAM",
            "speed": "⚡⚡⚡ Rápido",
            "quality": "⭐⭐⭐ Boa",
            "description": "Modelo open-source da EleutherAI, versátil para tarefas gerais."
        },
        "GPT-J-6B (6B parâmetros)": {
            "size": "6B parâmetros",
            "memory": "~12 GB RAM/VRAM",
            "speed": "⚡⚡ Médio",
            "quality": "⭐⭐⭐⭐ Muito Boa",
            "description": "Modelo mais poderoso da EleutherAI, ótimo equilíbrio qualidade/performance."
        },
        "Falcon-7B-Instruct (7B parâmetros)": {
            "size": "7B parâmetros",
            "memory": "~14 GB RAM/VRAM",
            "speed": "⚡⚡ Médio",
            "quality": "⭐⭐⭐⭐ Muito Boa",
            "description": "Modelo otimizado para instruções, desenvolvido pela TII."
        },
        "Llama-2-7B-Chat (7B parâmetros)": {
            "size": "7B parâmetros",
            "memory": "~14 GB RAM/VRAM",
            "speed": "⚡⚡ Médio",
            "quality": "⭐⭐⭐⭐⭐ Excelente",
            "description": "Modelo de alta qualidade da Meta, otimizado para conversação e instrução."
        },
        "Mistral-7B-Instruct (7B parâmetros)": {
            "size": "7B parâmetros",
            "memory": "~14 GB RAM/VRAM",
            "speed": "⚡⚡ Médio",
            "quality": "⭐⭐⭐⭐⭐ Excelente",
            "description": "Modelo state-of-the-art da Mistral AI, excelente para tarefas técnicas."
        }
    }
    
    info = model_info[selected_model_friendly]
    
    st.markdown("**Informações do Modelo:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **Tamanho:** {info['size']}")
        st.markdown(f"- **Memória necessária:** {info['memory']}")
    with col2:
        st.markdown(f"- **Velocidade:** {info['speed']}")
        st.markdown(f"- **Qualidade:** {info['quality']}")
    
    st.info(f"ℹ️ {info['description']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Botão para trocar modelo
    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        if st.button("🔄 Carregar Modelo Selecionado", use_container_width=True, type="primary", key="load_model_btn"):
            if selected_model_id == current_model_name and model_loaded:
                st.info("✅ Este modelo já está carregado!")
            else:
                with st.spinner(f"Carregando {selected_model_friendly}... Isso pode levar alguns minutos..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/change-model",
                            params={"model_name": selected_model_id},
                            timeout=300  # 5 minutos de timeout
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Modelo {selected_model_friendly} carregado com sucesso!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Erro ao carregar modelo: {response.status_code}")
                            st.code(response.text)
                    except Exception as e:
                        st.error(f"❌ Erro: {e}")
                        st.warning("⚠️ O carregamento pode demorar. Verifique os logs do backend.")
    
    with col_btn2:
        if st.button("🔄 Atualizar Status", use_container_width=True, key="refresh_model_status_btn"):
            st.rerun()
    
    # Avisos importantes
    st.markdown("#### ⚠️ Notas Importantes")
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    - **GPU Recomendada:** Modelos maiores (6B+) requerem GPU para melhor performance
    - **Memória:** Certifique-se de ter RAM/VRAM suficiente
    - **Primeiro Carregamento:** O download do modelo pode levar vários minutos
    - **Troca de Modelo:** O modelo anterior será descarregado da memória
    - **Fallback:** Se o carregamento falhar, o sistema usará geração por template
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------
# TAB 3 - Sistema
# --------------------------------------------------------------------
with tab3:
    st.markdown(f'### <span class="icon-inline">{icon("settings", 24)}</span> Status do Sistema', unsafe_allow_html=True)
    
    col_refresh, col_test = st.columns([3, 1])
    with col_refresh:
        if st.button("🔄 Atualizar Status", use_container_width=True, key="refresh_system_status_btn"):
            st.rerun()
    
    try:
        status_response = requests.get(f"{API_URL}/", timeout=5)
        if status_response.status_code == 200:
            status_data = status_response.json()
            backend_status = "🟢 Online"
            
            # Cards de métricas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='status-card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%)'>
                    <h3>Status</h3>
                    <p>✓ Online</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='status-card' style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'>
                    <h3>Versão</h3>
                    <p>{status_data.get('version', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='status-card' style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)'>
                    <h3>Device</h3>
                    <p>{status_data.get('device', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                openai_status = "✓" if status_data.get('openai_configured') else "✗"
                color = "#10b981" if status_data.get('openai_configured') else "#ef4444"
                st.markdown(f"""
                <div class='status-card' style='background: linear-gradient(135deg, {color} 0%, {color} 100%)'>
                    <h3>OpenAI</h3>
                    <p>{openai_status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Informações detalhadas
            st.markdown("#### 📋 Informações Detalhadas")
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            
            info_cols = st.columns(2)
            with info_cols[0]:
                st.metric("📚 Documentos Indexados", status_data.get('documents_indexed', 0))
            with info_cols[1]:
                persistence = status_data.get('persistence', {})
                persist_status = "✅ Ativo" if persistence.get('enabled') else "❌ Inativo"
                st.metric("💾 Sistema de Persistência", persist_status)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # JSON completo em expander
            with st.expander("🔍 Ver JSON Completo"):
                st.json(status_data)
                
        else:
            backend_status = "🔴 Offline"
            st.error(f"❌ Backend offline (Status: {status_response.status_code})")
    except Exception as e:
        backend_status = "🔴 Offline"
        st.error(f"❌ Não foi possível conectar ao backend: {e}")

# --------------------------------------------------------------------
# TAB 4 - Estatísticas
# --------------------------------------------------------------------
with tab4:
    st.markdown(f"### <span class='icon-inline'>{icon('chart', 24)}</span> Estatísticas do Corpus", unsafe_allow_html=True)
    
    col_refresh = st.columns([3, 1])
    with col_refresh[0]:
        if st.button("🔄 Atualizar Estatísticas", use_container_width=True, key="refresh_stats_btn"):
            st.rerun()
    
    try:
        corpus_stats = requests.get(f"{API_URL}/corpus-stats", timeout=5).json()
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Total de Vetores",
                value=corpus_stats.get('total_documents', 0),
                help="Número total de vetores no índice FAISS"
            )
        
        with col2:
            st.metric(
                label="📚 Documentos Únicos",
                value=corpus_stats.get('unique_documents', 0),
                help="Número de documentos únicos (sem contar chunks)"
            )
        
        with col3:
            st.metric(
                label="🧮 Dimensão",
                value=corpus_stats.get('embedding_dimension', 0),
                help="Dimensionalidade dos embeddings"
            )
        
        with col4:
            persist_icon = "✅" if corpus_stats.get('persisted') else "❌"
            st.metric(
                label="💾 Persistido",
                value=persist_icon,
                help="Status da persistência dos dados"
            )
        
        # Informações adicionais
        st.markdown("#### 📊 Detalhes do Corpus")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.json(corpus_stats)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar estatísticas: {e}")

# --------------------------------------------------------------------
# TAB 5 - Persistência
# --------------------------------------------------------------------
with tab5:
    st.markdown(f'### <span class="icon-inline">{icon("database", 24)}</span> Gestão de Persistência', unsafe_allow_html=True)
    
    # Botões de ação
    col_actions = st.columns(3)
    with col_actions[0]:
        if st.button("🔄 Atualizar Informações", use_container_width=True, key="refresh_persistence_btn"):
            st.rerun()
    with col_actions[1]:
        if st.button("💾 Salvar Corpus Manualmente", use_container_width=True, key="save_corpus_btn"):
            try:
                response = requests.post(f"{API_URL}/save-corpus", timeout=10)
                if response.status_code == 200:
                    st.success("✅ Corpus salvo com sucesso!")
                    st.rerun()
                else:
                    st.error(f"❌ Erro ao salvar: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Erro: {e}")
    
    with col_actions[2]:
        if st.button("🗑️ Limpar Corpus", use_container_width=True, type="secondary", key="clear_corpus_init_btn"):
            st.warning("⚠️ Esta ação irá limpar todo o corpus!")
            if st.button("✔️ Confirmar Limpeza", type="primary", key="confirm_clear_corpus_btn"):
                try:
                    response = requests.post(f"{API_URL}/clear-corpus", timeout=10)
                    if response.status_code == 200:
                        st.success("✅ Corpus limpo com sucesso!")
                        st.rerun()
                    else:
                        st.error(f"❌ Erro: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Erro: {e}")
    
    try:
        persist_stats = requests.get(f"{API_URL}/persistence-stats", timeout=5).json()
        
        # Status cards
        st.markdown("#### 📊 Status da Persistência")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            meta_status = "✅" if persist_stats.get('metadata_exists') else "❌"
            st.markdown(f"""
            <div class='status-card' style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'>
                <h3>Metadados</h3>
                <p>{meta_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            index_status = "✅" if persist_stats.get('index_exists') else "❌"
            st.markdown(f"""
            <div class='status-card' style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)'>
                <h3>Índice FAISS</h3>
                <p>{index_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='status-card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%)'>
                <h3>Backups</h3>
                <p>{persist_stats.get('num_backups', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='status-card' style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'>
                <h3>Documentos</h3>
                <p>{persist_stats.get('documents_in_memory', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Informações detalhadas
        st.markdown("#### 📋 Detalhes da Persistência")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        
        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.metric("💾 Tamanho dos Metadados", f"{persist_stats.get('metadata_size_kb', 0)} KB")
            st.metric("🗄️ Tamanho do Índice", f"{persist_stats.get('index_size_kb', 0)} KB")
        
        with detail_cols[1]:
            st.metric("📅 Última Modificação", persist_stats.get('last_modified', 'N/A'))
            st.metric("🔢 Vetores Indexados", persist_stats.get('vectors_indexed', 0))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # JSON completo
        with st.expander("🔍 Ver Dados Completos"):
            st.json(persist_stats)
        
        # Informações sobre backups
        if persist_stats.get('num_backups', 0) > 0:
            st.markdown("#### 📦 Sistema de Backups")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"""
            - ✅ **{persist_stats.get('num_backups', 0)} backups** disponíveis
            - 🔄 Backups automáticos antes de cada salvamento
            - 📅 Mantém os últimos 5 backups
            - 💾 Backups incluem metadados e índice FAISS
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar informações de persistência: {e}")

# --------------------------------------------------------------------
# TAB 6 - Referências (com upload)
# --------------------------------------------------------------------
with tab6:
    st.markdown(f"### <span class='icon-inline'>{icon('upload', 24)}</span> Upload de Referências", unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Envie arquivos de referência",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
        help="Formatos suportados: TXT, MD, PDF, DOCX"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} arquivo(s) selecionado(s)**")
        
        if st.button("📤 Enviar Referências", type="primary", use_container_width=True, key="upload_references_btn"):
            success_count = 0
            error_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            documents_to_upload = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processando {uploaded_file.name}...")
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    # Extract text from file
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension in ['txt', 'md']:
                        file_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                    elif file_extension == 'pdf':
                        pdf_reader = PdfReader(uploaded_file)
                        file_content = ""
                        for page in pdf_reader.pages:
                            file_content += page.extract_text() + "\n"
                    elif file_extension == 'docx':
                        docx_doc = DocxDocument(uploaded_file)
                        file_content = ""
                        for paragraph in docx_doc.paragraphs:
                            file_content += paragraph.text + "\n"
                    else:
                        file_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                    
                    doc_data = {
                        "id": f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}",
                        "title": uploaded_file.name,
                        "text": file_content,
                        "source": "Upload via Interface",
                        "year": str(datetime.now().year),
                        "authors": ["User Upload"]
                    }
                    
                    documents_to_upload.append(doc_data)
                    success_count += 1
                    
                except Exception as e:
                    st.error(f"❌ Erro ao processar {uploaded_file.name}: {e}")
                    error_count += 1
            
            # Upload all documents at once
            if documents_to_upload:
                try:
                    status_text.text("Enviando ao backend...")
                    upload_response = requests.post(
                        f"{API_URL}/upload-documents",
                        json=documents_to_upload,
                        timeout=60
                    )
                    
                    if upload_response.status_code == 200:
                        result = upload_response.json()
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.markdown(f"""
                        <div class='success-box'>
                            <b>✅ Upload concluído com sucesso!</b><br>
                            📄 Documentos processados: {result.get('documents_uploaded', 0)}<br>
                            🔢 Vetores adicionados: {result.get('vectors_added', 0)}<br>
                            📊 Total de vetores: {result.get('total_vectors', 0)}<br>
                            💾 Persistido automaticamente: {result.get('persisted', False)}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Erro ao enviar: {upload_response.status_code}")
                        st.code(upload_response.text)
                        
                except Exception as e:
                    st.error(f"❌ Erro de conexão: {e}")
                finally:
                    status_text.empty()
                    progress_bar.empty()
            
            if error_count > 0:
                st.warning(f"⚠️ {error_count} arquivo(s) com erro no processamento")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Lista de referências
    st.markdown(f"### <span class='icon-inline'>{icon('list', 24)}</span> Referências no Corpus", unsafe_allow_html=True)
    
    try:
        corpus_response = requests.get(f"{API_URL}/corpus-documents", timeout=5)
        if corpus_response.status_code == 200:
            corpus_data = corpus_response.json()
            documents = corpus_data.get("documents", [])
            
            if documents:
                # Summary
                st.markdown(f"""
                <div class='info-box'>
                    📚 <b>Total:</b> {corpus_data.get('total_unique_documents', 0)} documentos únicos | 
                    🔢 {corpus_data.get('total_chunks', 0)} chunks indexados
                </div>
                """, unsafe_allow_html=True)
                
                # Convert to DataFrame for better display
                df_data = []
                for doc in documents:
                    df_data.append({
                        "Título": doc['title'],
                        "Fonte": doc['source'],
                        "Ano": doc['year'],
                        "Tamanho": f"{doc['text_length']} chars",
                        "Chunks": doc['chunks'],
                        "Upload": doc.get('uploaded_at', 'N/A')
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("📭 Nenhuma referência disponível no corpus. Faça upload de documentos para começar!")
        else:
            st.error("❌ Erro ao carregar referências")
    except Exception as e:
        st.error(f"❌ Erro de conexão: {e}")

# --------------------------------------------------------------------
# TAB 7 - Documentos Gerados
# --------------------------------------------------------------------
with tab7:
    st.markdown(f"### <span class='icon-inline'>{icon('folder', 24)}</span> Documentos Gerados", unsafe_allow_html=True)
    
    try:
        reports_response = requests.get(f"{API_URL}/reports", timeout=5)
        if reports_response.status_code == 200:
            reports_data = reports_response.json()
            reports = reports_data.get("reports", [])
            
            if reports:
                st.markdown(f"""
                <div class='info-box'>
                    📄 <b>Total de documentos gerados:</b> {len(reports)}
                </div>
                """, unsafe_allow_html=True)
                
                # Convert to DataFrame
                df_data = []
                for rep in reports:
                    df_data.append({
                        "ID": rep['id'],
                        "Título": rep['title'],
                        "Data de Geração": rep['generated_at'],
                        "Tamanho": f"{rep['size_kb']} KB"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Opção de filtro por data (futuro)
                with st.expander("🔍 Filtros Avançados"):
                    st.info("Em breve: filtros por data, tamanho e tipo de documento")
                
            else:
                st.info("📭 Nenhum documento gerado ainda. Crie seu primeiro documento na aba 'Criar Documento'!")
        else:
            st.error("❌ Erro ao carregar documentos")
    except Exception as e:
        st.error(f"❌ Erro de conexão: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p><b>AutoReportAI v2.0</b> - Sistema Inteligente de Geração de Documentos com Persistência Automática</p>
    <p style='font-size: 0.875rem;'>Desenvolvido com ❤️ usando FastAPI, Streamlit, FAISS e OpenAI</p>
</div>
""", unsafe_allow_html=True)
