"""
Ranqueador de títulos: perfil de "home" (g1 / globo.com).

Saídas por título:
  1. Faixa (baixa / média / alta), a partir do ensemble de ranks (TF-IDF + tema+resíduo),
     com a taxa de home OBSERVADA em cada faixa na validação.
  2. Posição dentro do tema: chance estimada vs. média do tema.

AVISO: mede o perfil da matéria (assunto, gravidade, tom), não a qualidade da escrita.
A decisão de home depende de contexto editorial que o modelo não vê.

Uso no notebook:
    from ranqueador_home import treinar, carregar, analisa
    treinar(bd)                       # bd com colunas Title e Home_Gcom
    art = carregar()
    analisa(art, "Título novo aqui")
    analisa(art, ["Título 1", "Título 2"])   # devolve DataFrame
"""
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

ENC_NAME = "intfloat/multilingual-e5-base"
K_TEMAS = 30        # nº de temas (KMeans)
K_GRUPOS = 400      # grupos finos p/ manter quase-duplicados no mesmo fold
M_SUAV = 20         # suavização da taxa do tema
SEED = 42
CORTES = (0.50, 0.80)  # baixa: até p50 | média: p50-p80 | alta: acima de p80
NOMES = ["baixa", "média", "alta"]


# ---------- utilidades ----------
def limpa(s):
    return s.str.replace(r"\s*\|\s*G1\s*$", "", regex=True).str.strip()


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def embed(enc, titulos):
    return enc.encode(["query: " + t for t in titulos], batch_size=64,
                      normalize_embeddings=True, show_progress_bar=len(titulos) > 100)


def taxa_tema(tema, y, prior):
    g = pd.DataFrame({"t": tema, "y": y}).groupby("t")["y"].agg(["sum", "count"])
    return (g["sum"] + M_SUAV * prior) / (g["count"] + M_SUAV)


def feats_res(E, tema, centros, rate, prior):
    base = pd.Series(tema).map(rate).fillna(prior).values
    return np.hstack([E - centros[tema], logit(base)[:, None]])


def novo_tfidf():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, strip_accents="unicode"),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def pct(ref_ordenado, v):
    """Percentil (0-1) de v dentro da distribuição de referência."""
    return np.searchsorted(ref_ordenado, v, side="right") / len(ref_ordenado)


# ---------- treino ----------
def treinar(bd, col_titulo="Title", col_alvo="Home_Gcom", salvar="ranqueador_home.joblib"):
    titulos = limpa(bd[col_titulo]).values
    y = bd[col_alvo].values.astype(int)

    enc = SentenceTransformer(ENC_NAME)
    E = embed(enc, titulos)

    km = KMeans(K_TEMAS, random_state=SEED, n_init=10).fit(E)
    tema = km.labels_
    grupo = KMeans(K_GRUPOS, random_state=SEED, n_init=3).fit_predict(E)

    # --- scores fora da amostra (OOF), com quase-duplicados juntos no mesmo fold ---
    oof_tf = np.zeros(len(y))
    oof_rs = np.zeros(len(y))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for tr, te in cv.split(E, y, groups=grupo):
        prior = y[tr].mean()
        rate = taxa_tema(tema[tr], y[tr], prior)

        tf = novo_tfidf().fit(titulos[tr], y[tr])
        oof_tf[te] = tf.predict_proba(titulos[te])[:, 1]

        Xtr = feats_res(E[tr], tema[tr], km.cluster_centers_, rate, prior)
        Xte = feats_res(E[te], tema[te], km.cluster_centers_, rate, prior)
        rs = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr, y[tr])
        oof_rs[te] = rs.predict_proba(Xte)[:, 1]

    # --- ensemble de ranks, faixas e taxa observada por faixa ---
    s_tf, s_rs = np.sort(oof_tf), np.sort(oof_rs)
    ens = (pct(s_tf, oof_tf) + pct(s_rs, oof_rs)) / 2
    s_ens = np.sort(ens)
    cortes = np.quantile(ens, CORTES)
    faixa = np.digitize(ens, cortes)  # 0, 1, 2
    taxa_faixa = [float(y[faixa == i].mean()) for i in range(3)]
    n_faixa = [int((faixa == i).sum()) for i in range(3)]

    print(f"Taxa base de home: {y.mean():.1%}")
    print(f"PR-AUC (ensemble, OOF agrupado): {average_precision_score(y, ens):.3f}")
    print(f"ROC-AUC (ensemble, OOF agrupado): {roc_auc_score(y, ens):.3f}")
    for i, nome in enumerate(NOMES):
        print(f"  faixa {nome:6s}: {taxa_faixa[i]:.1%} de home "
              f"({taxa_faixa[i] / y.mean():.2f}x a média) | n={n_faixa[i]}")

    # --- modelos finais (base inteira) ---
    prior = y.mean()
    rate_all = taxa_tema(tema, y, prior)
    tf_final = novo_tfidf().fit(titulos, y)
    X_all = feats_res(E, tema, km.cluster_centers_, rate_all, prior)
    rs_final = LogisticRegression(max_iter=3000, C=0.5).fit(X_all, y)

    art = dict(km=km, rate_all=rate_all, prior=prior, tf=tf_final, rs=rs_final,
               s_tf=s_tf, s_rs=s_rs, s_ens=s_ens, cortes=cortes,
               taxa_faixa=taxa_faixa, taxa_base=float(y.mean()))
    joblib.dump(art, salvar)
    print(f"Salvo em {salvar}")
    return art


# ---------- uso ----------
def carregar(caminho="ranqueador_home.joblib"):
    art = joblib.load(caminho)
    art["enc"] = SentenceTransformer(ENC_NAME)
    return art


def analisa(art, titulos):
    unico = isinstance(titulos, str)
    lista = [titulos] if unico else list(titulos)
    t = pd.Series(lista)
    t = limpa(t).values

    E = embed(art["enc"], t)
    tema = art["km"].predict(E)
    X = feats_res(E, tema, art["km"].cluster_centers_, art["rate_all"], art["prior"])

    p_tf = art["tf"].predict_proba(t)[:, 1]
    p_rs = art["rs"].predict_proba(X)[:, 1]
    ens = (pct(art["s_tf"], p_tf) + pct(art["s_rs"], p_rs)) / 2
    faixa = np.digitize(ens, art["cortes"])
    media_tema = pd.Series(tema).map(art["rate_all"]).values

    df = pd.DataFrame({
        "titulo": lista,
        "faixa": [NOMES[i] for i in faixa],
        "taxa_obs_faixa": [f"{art['taxa_faixa'][i]:.1%}" for i in faixa],
        "percentil": (pct(art["s_ens"], ens) * 100).round(0).astype(int),
        "tema": tema,
        "media_do_tema": [f"{m:.1%}" for m in media_tema],
        "vs_media_do_tema": (p_rs / media_tema).round(2),
    })
    return df.iloc[0].to_dict() if unico else df