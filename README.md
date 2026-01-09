# Auditorium - Sistema Preditivo de Performance de Conteúdo

[![Status](https://img.shields.io/badge/Status-Em%20Produção-success)](https://github.com/mtharruda/Auditorium)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)](https://streamlit.io/)

> Sistema de Machine Learning que prediz audiência de artigos antes da publicação, utilizado em produção por equipe editorial de portal de notícias de grande tráfego.

---

##Visão Geral
**Auditorium** é uma aplicação web desenvolvida em Streamlit que utiliza **Random Forest Regressor** para predizer o número de pageviews de artigos de notícias com base em seus títulos. O sistema analisa características linguísticas, similaridade com conteúdo histórico e integra IA generativa para fornecer feedback editorial.

**Status:** Sistema em produção - Utilizado diariamente por equipe editorial

---

## Problema de Negócio

**Desafio:** Equipes editoriais publicam centenas de artigos diariamente sem previsibilidade de performance, dificultando:
- Priorização de recursos editoriais
- Otimização de timing de publicação
- Alocação eficiente de esforços da equipe

**Solução:** Sistema preditivo que estima audiência antes da publicação, permitindo decisões estratégicas baseadas em dados.

---

## Arquitetura e Funcionalidades

### Fluxo do Sistema

```
Título do Artigo → Feature Engineering → Modelo ML → Predição de Pageviews
                                    ↓
                              Gemini API → Feedback Editorial
                                    ↓
                         GitHub Integration → Histórico
```

### Principais Funcionalidades

1. **Predição de Audiência**
   - Modelo: Random Forest Regressor (Scikit-learn)
   - Input: Título do artigo
   - Output: Estimativa de pageviews em 24h
   - Métricas: MSE, R², MAE

2. **Análise de Similaridade**
   - Fuzzy matching com títulos históricos
   - Identifica conteúdo similar já publicado
   - Previne duplicação e canibalização de audiência

3. **Feedback Editorial com IA**
   - Integração com Google Gemini API
   - Sugestões de otimização de título
   - Análise de clarity, engagement e SEO

4. **Armazenamento de Histórico**
   - Integração com GitHub para versionamento
   - Registro de predições vs. performance real
   - Base de dados para retreinamento contínuo

---

## Stack Técnico

**Core:**
- **Python 3.9+**
- **Streamlit** - Interface web interativa
- **Scikit-learn** - Random Forest Regressor
- **Pandas & NumPy** - Manipulação de dados

**Integrações:**
- **Google Gemini API** - IA generativa para feedback editorial
- **GitHub API** - Versionamento e histórico de predições
- **FuzzyWuzzy** - Análise de similaridade de texto

**ML Pipeline:**
- Feature engineering de títulos
- TF-IDF vectorization
- Random Forest com hyperparameter tuning
- Cross-validation e métricas de performance

---

## Como Executar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Configuração

Criar arquivo `.env` com:

```bash
GEMINI_API_KEY=sua_chave_aqui
GITHUB_TOKEN=seu_token_aqui  # Opcional
```

### Execução

```bash
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`

---

## Resultados e Impacto

### Métricas do Modelo
- **R² Score:** ~0.72 (explica 72% da variância)
- **MAE:** ~8.500 pageviews
- **Performance:** Predições em tempo real (<1 segundo)

### Impacto em Produção
- ✅ **Utilizado diariamente** pela equipe editorial
- ✅ **Auxilia priorização** de conteúdo estratégico
- ✅ **Reduz incerteza** em decisões de publicação
- ✅ **Parte de estratégia** que resultou em **+50% de crescimento de audiência em 1 ano**

---

## Funcionalidades da Interface

### 1. Predição de Audiência
```
📝 Digite o título: "Nova descoberta sobre mudanças climáticas"
[PREVER AUDIÊNCIA]

📊 Resultado:
Pageviews estimados (24h): 42.350
Confiança do modelo: 78%
```

### 2. Análise de Similaridade
```
🔍 Títulos similares encontrados:
- "Cientistas descobrem nova evidência sobre clima" (85% similar)
- "Mudanças climáticas: o que mudou" (72% similar)
```

### 3. Feedback Editorial (Gemini)
```
💡 Sugestões de otimização:
✓ Título claro e direto
⚠ Considere adicionar número ou dado específico
✓ Boa otimização para SEO
```

---

##Estrutura do Projeto

```
Auditorium/
│
├── app.py                 # Aplicação Streamlit principal
├── requirements.txt       # Dependências Python
├── README.md             # Este arquivo
│
├── models/               # Modelos treinados (não versionados)
│   └── random_forest.pkl
│
└── data/                 # Dados históricos (não versionados)
    └── historical_data.csv
```

---

## Nota sobre Dados

Este repositório contém a estrutura e código da aplicação. Dados históricos e modelos treinados com informações proprietárias são mantidos em repositório privado por questões de confidencialidade.

**Demonstração:** Este código é funcional e pode ser adaptado para qualquer dataset de títulos + pageviews.

---

## Aprendizados Técnicos

### Feature Engineering
- Comprimento do título é preditor significativo
- Presença de números aumenta engajamento
- Palavras-chave específicas correlacionam com audiência

### Modelo
- Random Forest superou modelos lineares e XGBoost
- Ensemble methods são ideais para dados textuais
- Overfitting controlado via cross-validation

### Produção
- Integração Streamlit permite adoção rápida por equipe não-técnica
- Feedback loop (predição → resultado real) essencial para retreinamento
- API de IA generativa complementa análise quantitativa

---

## Exemplo de Uso

```python
# Carregar modelo treinado
import joblib
model = joblib.load('models/random_forest.pkl')

# Predizer audiência
titulo = "Nova descoberta sobre mudanças climáticas"
features = extract_features(titulo)
predicao = model.predict([features])

print(f"Pageviews estimados: {predicao[0]:,.0f}")
```
---

## Roadmap Futuro

- [ ] Adicionar classificação multi-classe (alta/média/baixa)
- [ ] Incorporar features de timing (dia da semana, hora)
- [ ] Dashboard de performance do modelo
- [ ] API REST para integração com CMS
- [ ] A/B testing de títulos em tempo real

---
<div align="center">
Made with ❤️ and ☕ by [Matheus Arruda](https://github.com/mtharruda)
</div>
