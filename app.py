'''
Código desenvolvido para predizer a audiência de matérias jornalísticas a partir da análise do título fornecido pelo editor de texto (usuário).
A predição ocorre pr meio de aprendizado de máquinas, com uso da biblioteca SciKit Learn, em um modelo de REGRESSÃO LINEAR. 
Na segunda etapa, o algoritmo estabelece conexão com o Gemini, modelo de Inteligência Artificial generativa do Google. E realiza uma requisição para analisar e fornecer dicas de engajamento ao conteúdo.
Código por Matheus Arruda, março de 2025
'''

#BIBLIOTECAS
import streamlit as st #Framework para disponibilizar na web o app 
#Análise e tratamento de dados
import pandas as pd
import numpy as np
import time
from datetime import datetime

#Manipulação de diretórios e arquivos (usaremos para administrar o modelo - previamente carregado)
import os
import joblib
import gdown
import base64
#Gen AI
import google.generativeai as genai
from fuzzywuzzy import fuzz
#Lib do Github - usaremos para salvar todos os títulos imputados pelo usuário. 
from github import Github, GithubException
# ======================================================
# 1. CONFIGURAÇÕES INICIAIS
# ======================================================

st.set_page_config(
    page_title="Auditorium",
    page_icon="logo.png"
)

# ======================================================
# CARREGAMENTO DE DADOS E MODELO
# ======================================================
# --- Carregar Segredos ---
try:
    GITHUB_TOKEN = st.secrets["github"]["token"]
    REPO_NAME = st.secrets["github"]["repo"]
    FILE_PATH = st.secrets["github"]["filepath"]
    BRANCH = st.secrets["github"].get("branch", "main")
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError as e:
    st.error(f"Erro de Configuração: {e}. Verifique .streamlit/secrets.toml")
    st.stop()

# Carregar dados para comparação de similaridade
@st.cache_data
def load_data():
    if os.path.exists('bd_modelpred.csv'):
        return pd.read_csv('bd_modelpred.csv', sep=',')
    return pd.DataFrame() # Retorna vazio se não existir para não quebrar
bd = load_data()

# --- FUNÇÃO VITAL: PREPARA OS DADOS PARA O MODELO ---
# O modelo Pipeline espera receber Data, Hora e User Need além do Título
def enriquecer_dataset(df):
    df_novo = df.copy()
    coluna_data = 'Publicação'
    
    # Tratamento de Texto
    titulos = df_novo['Matéria'].astype(str)
    df_novo['tam_titulo'] = titulos.str.len()
    df_novo['num_palavras'] = titulos.apply(lambda x: len(x.split()))
    df_novo['tem_interrogacao'] = titulos.apply(lambda x: 1 if '?' in x else 0)
    df_novo['tem_exclamacao'] = titulos.apply(lambda x: 1 if '!' in x else 0)
    df_novo['tem_numero'] = titulos.apply(lambda x: 1 if any(c.isdigit() for c in x) else 0)
    
    # Tratamento de Data
    if coluna_data in df_novo.columns:
        df_novo[coluna_data] = pd.to_datetime(df_novo[coluna_data], errors='coerce')
        df_novo['dia_semana'] = df_novo[coluna_data].dt.dayofweek
        df_novo['hora'] = df_novo[coluna_data].dt.hour
        df_novo['eh_fim_de_semana'] = df_novo['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
        # Para novos posts, dias no ar é 0
        df_novo['dias_no_ar'] = 0 
    return df_novo

# --- DOWNLOAD E CARREGAMENTO DO MODELO ---
@st.cache_resource(show_spinner=False)
def load_model_from_drive():
    FILE_ID = "1PRruXA-oB_tR-dFG-o2_Oad8heHO-unz"
    OUTPUT_FILE = "modelo_classificacao_v2.joblib"
    
    # URL de download direto do GDrive
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    
    # Baixa apenas se não existir
    if not os.path.exists(OUTPUT_FILE):
        with st.spinner("Preparando modelo de inteligência..."):
            gdown.download(url, OUTPUT_FILE, quiet=False)
            
    # Carrega o Pipeline completo
    try:
        return joblib.load(OUTPUT_FILE)
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}.")
        return None

model_pipeline = load_model_from_drive()

# ======================================================
# INTERFACE DE ENTRADA (INPUTS)
# ======================================================

# O novo modelo precisa dessas informações além do título
col1, col2 = st.columns(2)
with col1:
    data_pub = st.date_input("Data da publicação", datetime.now())
with col2:
    hora_pub = st.time_input("Horário", datetime.now().time())

user_need = st.selectbox(
    "User Need",
    ("Informar", "Contextualizar", "Ensinar", "Entreter", "Inspirar", "Acompanhar assuntos em alta")
)

novo_titulo = st.text_input("Digite o título da notícia")

# ======================================================
# 4. FUNÇÕES AUXILIARES (GitHub, Gemini, Validação)
# ======================================================

def validar_titulo(titulo):
    t = titulo.strip()
    if not t: return False, "Título vazio."
    if len(t) < 10: return False, "Título muito curto."
    return True, ""

def analisar_qualidade_titulo(titulo):
    # (Mantive sua função original de análise de regras)
    feedback = []
    t = titulo.strip()
    if len(t) < 50: feedback.append(f"⚠️ Curto para SEO ({len(t)} chars)")
    elif len(t) > 70: feedback.append(f"⚠️ Longo para SEO ({len(t)} chars)")
    else: feedback.append("✅ Tamanho ideal SEO")
    if "?" in t: feedback.append("✅ Pergunta gera engajamento")
    return feedback

def ajuda_editor(titulo):
    # Atualizei o prompt para considerar a classe prevista
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    try:
        prompt = f"""
        Analise o título: '{titulo.strip()}' (Contexto: G1).
        1. Dê nota 0-10 para SEO/Atratividade.
        2. Dê nota 0-10 para Google Discover.
        3. Sugira duas alternativas profissionais e otimizadas, sendo uma para SEO e outra para o Google Discover, baseada no contexto do g1. 
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erro Gemini: {str(e)}"

# Função GitHub (Mantida a sua lógica)
def append_to_github_txt(token, repo_name, file_path, text_to_append, audience_to_append, commit_message, branch="main"):
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        try:
            contents = repo.get_contents(file_path, ref=branch)
            decoded = base64.b64decode(contents.content).decode("utf-8")
            sha = contents.sha
        except GithubException:
            decoded = ""
            sha = None
        
        # Salva o resultado formatado
        new_content = decoded.strip() + "\n" + text_to_append + "|" + audience_to_append + "\n"
        
        if sha:
            repo.update_file(file_path, commit_message, new_content.encode("utf-8"), sha, branch=branch)
        else:
            repo.create_file(file_path, commit_message, new_content.encode("utf-8"), branch=branch)
        return True
    except Exception as e:
        st.error(f"Erro Github: {e}")
        return False

# ======================================================
# INTERFACE
# ======================================================
st.title("Previsão de audiência")
with st.sidebar:
    st.info("Este modelo utiliza um sistema de classificação pra predizer a audiência de uma reportagem com base no título, user need e dia da publicação")
    st.link_button("Quer saber mais? Leia a documentação", "https://drive.google.com/file/d/1TRwzGB2xoF-HlEko_otSY_7s1Lh8rNDw")


# =======================================================
# INPUT DO USUÁRIO
# =======================================================
if st.button("Analisar título", type="primary"):
    if not novo_titulo.strip():
        st.warning("Digite um título.")
    elif model_pipeline is None:
        st.error("Modelo não carregado. Contate o administrador.")
    else:
        # 1. Preparar os dados para o Pipeline
        data_str = f"{data_pub} {hora_pub}"
        input_df = pd.DataFrame({
            'Matéria': [novo_titulo],
            'User Need': [user_need],
            'Publicação': [data_str]
        })
        
        # 2. Aplicar Engenharia de Features
        try:
            input_processed = enriquecer_dataset(input_df)
            
            # 3. Predição (Classificação)
            predicao_classe = model_pipeline.predict(input_processed)[0]
            probs = model_pipeline.predict_proba(input_processed)[0]
            classes = model_pipeline.classes_
            
            # Pega a confiança da classe vencedora
            confianca_dict = dict(zip(classes, probs))
            confianca_valor = confianca_dict[predicao_classe] * 100
            
            # --- EXIBIÇÃO ---
            st.divider()
            col_res1, col_res2 = st.columns([1, 2])
            
           
            
            with col_res1:
                cor = "green" if predicao_classe == "Alta" else "orange" if predicao_classe == "Média" else "red"
                st.markdown(f"#### Eficiência: :{cor}[{predicao_classe}]", help = 'Baixa: menos de 100 views | Média: até 1000 views | Alta: mais de 2000 views')
                st.metric("Confiança", f"{confianca_valor:.1f}%", help="Probabilidade matemática da previsão se concretizar")

                if confianca_valor < 60:
                    st.warning("O modelo não está confiante quanto à predição. O resultado pode ser imprevisível. Considere alterar o título, User Need ou data de publicação.")
                               
            with col_res2:
                # Gráfico de barras das probabilidades
                chart_data = pd.DataFrame({"Confiança": probs}, index=classes)
                st.bar_chart(chart_data)

            # Salvar no GitHub
            commit_msg = f"App Input: {predicao_classe}"
            append_to_github_txt(GITHUB_TOKEN, REPO_NAME, FILE_PATH, 
                                 f"{novo_titulo} ({user_need})", 
                                 predicao_classe, 
                                 commit_msg, BRANCH)
            
            # Feedback de Regras
            regras = analisar_qualidade_titulo(novo_titulo)
            with st.expander("Análise técnica"):
                for r in regras: st.write(r)
            
            # Gemini
            st.subheader("Análise da IA")
            with st.spinner("Conversando com Gemini..."):
                analise_ia = ajuda_editor(novo_titulo)
                st.write(analise_ia)

        except Exception as e:
            st.error(f"Erro no processamento: {e}")
            st.info("Verifique se as bibliotecas (scikit-learn) do seu ambiente local são compatíveis com a versão do Colab.")

  #FIM DO CÓDIGO
