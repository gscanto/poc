import streamlit as st
import requests
import json
from datetime import datetime
import time

# URL do backend FastAPI
API_URL = "http://localhost:8000"

# Configuração da página
st.set_page_config(
    page_title="AutoReportAI - Gerador Inteligente",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    .reference-item {
        background-color: #f8fafc;
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #667eea;
    }
    .section-badge {
        display: inline-block;
        background-color: #e0e7ff;
        color: #4f46e5;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<p class="main-header">📊 AutoReportAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Gerador Inteligente de Documentos Técnicos com IA</p>', unsafe_allow_html=True)

# Verificar status do backend
try:
    status_response = requests.get(f"{API_URL}/", timeout=2)
    if status_response.status_code == 200:
        status_data = status_response.json()
        backend_status = "🟢 Online"
        backend_color = "#10b981"
    else:
        backend_status = "🟡 Instável"
        backend_color = "#f59e0b"
except:
    backend_status = "🔴 Offline"
    backend_color = "#ef4444"

# Info cards no topo
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 10px; text-align: center;">
        <h3 style="color: white; margin: 0; font-size: 1rem;">Status do Backend</h3>
        <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0 0 0; font-weight: 600;">{backend_status}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    docs_count = status_data.get('documents_indexed', 0) if 'status_data' in locals() else 0
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.2rem; border-radius: 10px; text-align: center;">
        <h3 style="color: white; margin: 0; font-size: 1rem;">Documentos Indexados</h3>
        <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0 0 0; font-weight: 600;">{docs_count}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    device = status_data.get('device', 'N/A') if 'status_data' in locals() else 'N/A'
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.2rem; border-radius: 10px; text-align: center;">
        <h3 style="color: white; margin: 0; font-size: 1rem;">Dispositivo</h3>
        <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0 0 0; font-weight: 600;">{device.upper()}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- ÁREA PRINCIPAL ---
tab1, tab2, tab3 = st.tabs(["📄 Criar Documento", "📊 Estatísticas", "❓ Ajuda"])

with tab1:
    st.markdown("### 📝 Informações Básicas")
    
    title = st.text_input(
        "Título do documento",
        "Documento Técnico Automático",
        help="Defina um título descritivo para seu documento"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 Contexto do Documento")
    st.markdown("Descreva o tema e objetivo do seu documento")
    
    context = st.text_area(
        "",
        placeholder="Exemplo: Este documento analisa o desempenho de modelos de aprendizado de máquina aplicados à classificação de defeitos em processos industriais, com foco em redes neurais convolucionais e técnicas de aumento de dados...",
        height=180,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📑 Estrutura do Documento")
    st.markdown("Defina as seções que comporão seu documento")
    
    col_preset, col_custom = st.columns([1, 2])
    
    with col_preset:
        st.markdown("**Modelos pré-definidos:**")
        preset = st.radio(
            "",
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
            default_sections = ["Introdução", "Metodologia", "Resultados", "Discussão", "Conclusão"]
        
        sections = st.text_area(
            "",
            value="\n".join(default_sections),
            height=150,
            label_visibility="collapsed",
            help="Uma seção por linha"
        )
    
    # Preview das seções
    section_list = [s.strip() for s in sections.splitlines() if s.strip()]
    if section_list:
        st.markdown("**Preview das seções:**")
        preview_html = " ".join([f'<span class="section-badge">{s}</span>' for s in section_list])
        st.markdown(preview_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Nota sobre configurações automáticas
    # st.markdown("""
    # <div class="info-box">
    #     <strong>ℹ️ Configurações Automáticas</strong><br>
    #     Este sistema usa configurações otimizadas automaticamente:
    #     • Estilo: Técnico e formal<br>
    #     • Formato de referências: IEEE<br>
    #     • Recuperação inteligente de documentos relevantes
    # </div>
    # """, unsafe_allow_html=True)
    
    # Botão de geração
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_btn = st.button("🚀 Gerar Documento", type="primary", use_container_width=True)
    
    if generate_btn:
        if not context.strip():
            st.error("⚠️ Por favor, preencha o contexto antes de gerar.")
        elif not section_list:
            st.error("⚠️ Adicione pelo menos uma seção ao documento.")
        else:
            # Progress bar animado
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Preparando requisição...")
            progress_bar.progress(10)
            time.sleep(0.2)
            
            try:
                payload = {
                    "title": title,
                    "context": context,
                    "sections": section_list
                }
                
                status_text.text(f"🤖 Gerando {len(section_list)} seções com IA...")
                progress_bar.progress(20)
                
                # Fazer requisição SEM timeout - aguardar o tempo necessário
                status_text.text("⏳ Processando... Isso pode levar alguns minutos dependendo do tamanho do documento.")
                progress_bar.progress(30)
                
                response = requests.post(
                    f"{API_URL}/generate-report", 
                    json=payload,
                    timeout=None  # SEM TIMEOUT - aguarda indefinidamente
                )
                
                progress_bar.progress(90)
                
                if response.status_code == 200:
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    data = response.json()
                    
                    # Success message
                    st.markdown(f"""
                    <div class="success-box">
                        <h3 style="margin: 0 0 0.5rem 0; color: #047857;">✅ Documento Gerado com Sucesso!</h3>
                        <p style="margin: 0; color: #065f46;">
                            <strong>ID:</strong> {data['report_id']} | 
                            <strong>Tempo:</strong> {data['generation_time']:.2f}s | 
                            <strong>Tokens:</strong> {data['tokens_used']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tabs para visualização
                    view_tab1, view_tab2, view_tab3 = st.tabs(["📄 Documento", "📚 Referências", "📊 Métricas"])
                    
                    with view_tab1:
                        st.markdown("### Conteúdo Gerado")
                        st.markdown(data['content'])
                        
                        # Botão de download
                        md_bytes = data["content"].encode('utf-8')
                        st.download_button(
                            label="⬇️ Baixar Documento (Markdown)",
                            data=md_bytes,
                            file_name=f"{data['report_id']}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    
                    with view_tab2:
                        if data.get("references"):
                            st.markdown(f"### 📚 {len(data['references'])} Referências Recuperadas")
                            for i, ref in enumerate(data["references"], 1):
                                score = ref.get('score', 0)
                                st.markdown(f"""
                                <div class="reference-item">
                                    <strong>[{i}] {ref.get('title', 'Sem título')}</strong><br>
                                    <small>📖 {ref.get('source', 'N/A')} • 📅 {ref.get('year', 'N/A')} • 🎯 Score: {score:.3f}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("Nenhuma referência foi recuperada para este documento.")
                    
                    with view_tab3:
                        met_col1, met_col2, met_col3 = st.columns(3)
                        with met_col1:
                            st.metric("Tempo de Geração", f"{data['generation_time']:.2f}s")
                        with met_col2:
                            st.metric("Tokens Utilizados", data['tokens_used'])
                        with met_col3:
                            st.metric("Seções Geradas", len(section_list))
                        
                        st.markdown("---")
                        st.markdown("**Detalhes da Configuração:**")
                        config_info = f"""
                        - **Estilo:** Technical (Técnico e Formal)
                        - **Formato de Referências:** IEEE
                        - **Top-K Documentos:** 6
                        - **Recuperação Ativa:** ✅ Sim
                        """
                        st.markdown(config_info)
                
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Erro ao gerar documento: {response.status_code}")
                    with st.expander("Ver detalhes do erro"):
                        st.code(response.text)
            
            except requests.exceptions.RequestException as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Erro de conexão: {str(e)}")
                st.info("💡 Verifique se o backend está rodando em http://localhost:8000")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Erro inesperado: {str(e)}")

with tab2:
    st.markdown("### 📊 Estatísticas do Sistema")
    
    if st.button("🔄 Atualizar Estatísticas", use_container_width=True):
        try:
            status = requests.get(f"{API_URL}/").json()
            corpus_stats = requests.get(f"{API_URL}/corpus-stats").json()
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.markdown("#### Status Geral")
                st.json(status)
            
            with col_stat2:
                st.markdown("#### Corpus de Documentos")
                st.json(corpus_stats)
            
        except Exception as e:
            st.error(f"Erro ao obter estatísticas: {e}")
    else:
        st.info("👆 Clique no botão acima para carregar as estatísticas do sistema")

with tab3:
    st.markdown("### ❓ Como Usar o AutoReportAI")
    
    st.markdown("""
    #### 🚀 Guia Rápido
    
    1. **Defina o título** do seu documento
    
    2. **Descreva o contexto** do documento:
       - Seja claro e específico sobre o tema
       - Inclua objetivos e escopo
       - Mencione técnicas ou metodologias relevantes
    
    3. **Escolha ou personalize as seções**:
       - Use modelos pré-definidos (Acadêmico, Técnico, Executivo)
       - Ou personalize as seções conforme sua necessidade
       - Uma seção por linha
       - A ordem será mantida no documento final
    
    4. **Gere e baixe** seu documento:
       - Clique em "Gerar Documento"
       - Aguarde o processamento (pode levar alguns minutos)
       - Visualize o conteúdo gerado
       - Revise as referências utilizadas
       - Baixe em formato Markdown
    
    ---
    
    #### 💡 Dicas de Uso
    
    - **Contexto detalhado**: Quanto mais informações você fornecer no contexto, melhor será o resultado
    - **Seções bem definidas**: Use títulos claros e específicos para cada seção
    - **Aguarde pacientemente**: A geração de documentos com IA pode levar tempo, especialmente para documentos longos
    
    #### ⚙️ Configurações Automáticas
    
    O sistema usa configurações otimizadas automaticamente:
    - **Estilo de escrita**: Técnico e formal, ideal para documentos acadêmicos e profissionais
    - **Formato de referências**: IEEE (padrão internacional para documentos técnicos)
    - **Recuperação de documentos**: Busca automática de 6 documentos mais relevantes
    - **IA avançada**: Utiliza modelos de linguagem de última geração
    
    #### 🔧 Requisitos
    
    - Backend FastAPI rodando em `http://localhost:8000`
    - Modelos de IA carregados (embedding + LLM)
    - Corpus de documentos indexado
    
    #### 📞 Solução de Problemas
    
    Se encontrar problemas:
    - ✅ Verifique se o backend está online (card no topo da página)
    - ✅ Revise os logs do FastAPI no terminal
    - ✅ Certifique-se de que há espaço suficiente em disco
    - ✅ Verifique a conexão de internet (para download de modelos)
    
    #### 🎯 Exemplos de Uso
    
    **Documento Acadêmico:**
    - Contexto: "Análise comparativa de algoritmos de deep learning para detecção de fraudes em transações financeiras"
    - Seções: Resumo, Introdução, Revisão da Literatura, Metodologia, Resultados, Discussão, Conclusão
    
    **Relatório Técnico:**
    - Contexto: "Implementação de sistema de monitoramento em tempo real usando IoT e edge computing"
    - Seções: Sumário Executivo, Especificações Técnicas, Arquitetura, Implementação, Testes
    
    **Apresentação Executiva:**
    - Contexto: "Proposta de adoção de MLOps para otimizar pipeline de machine learning"
    - Seções: Sumário Executivo, Contexto, Análise, Recomendações, Próximos Passos
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    <p style="margin: 0;">AutoReportAI v1.0</p>
</div>
""", unsafe_allow_html=True)
