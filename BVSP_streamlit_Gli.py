# app_streamlit_bvsp_svm.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import altair as alt


# =========================
# Config
# =========================
st.set_page_config(page_title="PAINEL BVSP + SVM", layout="wide")
st.title("📈 Painel BVSP (IBOV) + Previsão com SVM")
st.caption("Deploy + Monitoramento do modelo (FIAP - Tech Challenge Fase 4).")

with st.expander("📌 Instruções de Uso"):
    st.write(
        """
**O que este app faz (clarinho):**
- **Gráfico de preços**: vem do **Yahoo Finance (ao vivo)** para o ticker que você digitar.
- **Previsão e métricas do modelo SVM**: usam o **dataset tratado do notebook** (`dados_tratados_data_correta.csv`).
  - Isso é intencional para **reproduzir a lógica/métricas do notebook** (ex.: acurácia 0,867).
- Para não confundir:
  - **Ticker do Yahoo** = apenas indicadores e gráfico.
  - **Modelo SVM** = previsão calculada no dataset tratado (não depende do ticker).
"""
    )


# =========================
# Defaults
# =========================
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/paulopetrillo/"
    "FIAP_TECH_CHALENGE_04/main/dados_tratados_data_correta.csv"
)


# =========================
# Helpers / Loaders
# =========================
@st.cache_resource
def load_model_and_features():
    """Espera joblib como dict: {'model': ..., 'feature_cols': [...], 'threshold': ... (opcional)}"""
    path = Path(__file__).parent / "models" / "svm_bvsp.joblib"
    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise TypeError("O arquivo joblib não é um dict. Salve como {'model':..., 'feature_cols':...}.")

    if "model" not in artifact:
        raise KeyError("Não achei a chave 'model' dentro do artifact (.joblib).")

    if "feature_cols" not in artifact:
        raise KeyError("Não achei a chave 'feature_cols' dentro do artifact (.joblib).")

    model = artifact["model"]
    features = list(artifact["feature_cols"])
    threshold = float(artifact.get("threshold", 0.5))  # opcional

    return model, features, threshold


@st.cache_data(ttl=3600)
def load_dataset(local_first: bool, data_url: str) -> tuple[pd.DataFrame, str]:
    """
    Retorna:
      df, dataset_source  -> 'local_root' | 'local_data' | 'url'
    """
    root = Path(__file__).parent / "dados_tratados_data_correta.csv"
    data_dir = Path(__file__).parent / "data" / "dados_tratados_data_correta.csv"

    if local_first:
        if root.exists():
            return pd.read_csv(root), "local_root"
        if data_dir.exists():
            return pd.read_csv(data_dir), "local_data"

    return pd.read_csv(data_url), "url"

@st.cache_data(ttl=3600)
def fetch_market_history(ticker_symbol: str, years: int) -> pd.DataFrame:
    """Histórico do Yahoo até o último pregão disponível."""
    emp = yf.Ticker(ticker_symbol)
    hist = emp.history(period=f"{years}y")
    return hist


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Tenta colocar Data/Date como índice datetime (se existir)."""
    df = df.copy()
    for col in ["Date", "date", "Data", "DATA"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).sort_values(col).set_index(col)
            return df
    return df


def align_X_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Numérico, ordem de colunas do treino (se existir), remove NaNs."""
    X = X.copy().apply(pd.to_numeric, errors="coerce")

    # Se Pipeline, tenta usar feature_names_in_ (normalmente do scaler)
    if isinstance(model, Pipeline):
        for step_name, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                expected = list(step.feature_names_in_)
                missing = [c for c in expected if c not in X.columns]
                if missing:
                    raise ValueError(f"Faltam colunas esperadas no pipeline ({step_name}): {missing}")
                X = X.reindex(columns=expected)
                break

    X = X.loc[X.notna().all(axis=1)]
    return X


def append_log(row: dict) -> None:
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / "usage_log.csv"

    df_row = pd.DataFrame([row])
    if log_path.exists():
        df_row.to_csv(log_path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False, encoding="utf-8")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def fmt_brl(x: float) -> str:
    # 12345.67 -> "R$ 12.345,67"
    s = f"{x:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

def fmt_index_pts(x: float) -> str:
    """
    Formata número como pontos de índice no padrão BR:
    163154.12 -> "163.154"
    (arredonda para inteiro)
    """
    n = int(round(x))
    s = f"{n:,}".replace(",", ".")
    return s


def fmt_price_by_ticker(ticker_symbol: str, x: float) -> str:
    """
    Se for Ibovespa (^BVSP) -> pontos (sem R$)
    Senão -> moeda BRL (R$)
    """
    if ticker_symbol == "^BVSP":
        return fmt_index_pts(x)
    return fmt_brl(x)


# =========================
# Sidebar (controles)
# =========================
st.sidebar.header("⚙️ Configurações")

ticker = st.sidebar.text_input("Ticker (use ^BVSP para índice)", "^BVSP").strip().upper()
ticker_symbol = f"{ticker}.SA" if ticker != "^BVSP" else "^BVSP"

years = st.sidebar.slider("Histórico (anos) — Yahoo", 1, 15, 10)
n_test = st.sidebar.slider("Janela (últimos pregões) — teste/métricas (dataset tratado)", 5, 252, 30)

root_path = Path(__file__).parent / "dados_tratados_data_correta.csv"
data_path = Path(__file__).parent / "data" / "dados_tratados_data_correta.csv"

local_data_first = st.sidebar.checkbox(
    "Usar dataset local (arquivo do projeto)",
    value=True
)

data_url = st.sidebar.text_input("Fallback URL (raw) do dataset tratado", DEFAULT_DATA_URL)

enable_logging = st.sidebar.checkbox("Salvar log de uso (CSV)", value=True)
show_monitoring = st.sidebar.checkbox("Mostrar painel de monitoramento", value=True)

# Botão de recarregar (limpa cache)
if st.sidebar.button("🔄 Recarregar dados (limpar cache)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# Debug: o que o usuário marcou + se há arquivo local disponível
local_available = root_path.exists() or data_path.exists()
st.sidebar.markdown(
    f"**Preferência marcada:** {'LOCAL' if local_data_first else 'URL'}"
    f"<br>**Local disponível:** {'SIM' if local_available else 'NÃO'}",
    unsafe_allow_html=True
)


# =========================
# Carregar modelo
# =========================
try:
    model, FEATURES, threshold = load_model_and_features()
    st.success(f"Modelo carregado: {type(model).__name__} | #features: {len(FEATURES)}")
except Exception as e:
    st.error("Falha ao carregar o modelo (models/svm_bvsp.joblib).")
    st.exception(e)
    st.stop()


# =========================
# Seção A — Mercado (Yahoo)
# =========================
st.subheader("📊 Mercado (Yahoo Finance) — **usa o ticker escolhido**")

with st.spinner("Baixando histórico do Yahoo Finance..."):
    hist = fetch_market_history(ticker_symbol, years)

if hist.empty:
    st.warning("Sem histórico retornado pelo Yahoo Finance para esse ticker.")
else:
    last_dt = hist.index.max()
    st.caption(f"Última data disponível no Yahoo (histórico): **{last_dt.date()}**")

    colA, colB, colC = st.columns(3)
    colA.metric("Último Close (Yahoo)", fmt_price_by_ticker(ticker_symbol, float(hist["Close"].iloc[-1])))
    colB.metric("Máximo (período)", fmt_price_by_ticker(ticker_symbol, float(hist["Close"].max())))
    colC.metric("Mínimo (período)", fmt_price_by_ticker(ticker_symbol, float(hist["Close"].min())))

    price_df = hist.reset_index()[["Date", "Close"]].dropna()
    chart = (
        alt.Chart(price_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Data"),
            y=alt.Y("Close:Q", title=f"Fechamento — {ticker_symbol} (Yahoo)"),
            tooltip=["Date:T", "Close:Q"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# pegar indicadores do Yahoo (último e anterior) para exibir
yahoo_last_close = None
yahoo_prev_close = None
yahoo_pct = None

if not hist.empty and "Close" in hist.columns and len(hist) >= 2:
    yahoo_last_close = float(hist["Close"].iloc[-1])
    yahoo_prev_close = float(hist["Close"].iloc[-2])
    yahoo_pct = (yahoo_last_close / yahoo_prev_close - 1.0) * 100.0

st.divider()


# =========================
# Seção B — Modelo (dataset tratado)
# =========================
st.warning(
    "⚠️ **Esta seção NÃO usa o ticker para prever.** "
    "A previsão/métricas vêm do **dataset tratado do notebook** para reproduzir seu experimento."
)

# carregar dataset tratado (uma vez só)
try:
    df_raw, dataset_source = load_dataset(local_data_first, data_url)
    df = ensure_datetime_index(df_raw)

    used_local = dataset_source.startswith("local")

    # ✅ debug na sidebar (a fonte REAL)
    st.sidebar.caption(f"Fonte carregada (REAL): {dataset_source.upper()}")

except Exception as e:
    st.error("Falha ao carregar o dataset tratado.")
    st.exception(e)
    st.stop()

# preview (reaproveita o que já foi carregado)
with st.expander("📄 Ver preview do dataset tratado"):
    st.caption(f"Fonte: **{dataset_source.upper()}**")
    st.dataframe(df_raw.tail(15), use_container_width=True)

st.subheader("🧠 Previsão (últimos N registros do dataset tratado)")


# validações
if "Target" not in df_raw.columns and "Target" not in df.columns:
    st.error("O dataset precisa ter a coluna **Target** (como no notebook).")
    st.stop()

# y
y_all = (df["Target"] if "Target" in df.columns else df_raw["Target"]).astype(int).copy()

# X
missing_feats = [c for c in FEATURES if c not in df_raw.columns and c not in df.columns]
if missing_feats:
    st.error("Faltam features no dataset tratado (não bate com o modelo).")
    st.write(missing_feats)
    st.stop()

base_df = df if all(c in df.columns for c in FEATURES) else df_raw
X_all = base_df[FEATURES].copy()

try:
    X_all = align_X_to_model(X_all, model)
    y_all = y_all.loc[X_all.index]  # alinha após dropna
except Exception as e:
    st.error("Falha ao alinhar X para o modelo (ordem/colunas/números).")
    st.exception(e)
    st.stop()

# UI: botão
btn = st.button("Gerar previsão", type="primary")

if btn:
    X_test = X_all.tail(int(n_test))
    y_test = y_all.tail(int(n_test))

    if X_test.empty:
        st.warning("Sem dados suficientes para prever (X_test vazio).")
        st.stop()

    # Prever
    y_pred = model.predict(X_test)

    # Score (0-1) opcional: decision_function -> sigmoid
    score01 = None
    if hasattr(model, "decision_function"):
        try:
            dec = model.decision_function(X_test.iloc[[-1]])
            score01 = float(sigmoid(float(np.ravel(dec)[0])))
        except Exception:
            score01 = None

    # resultado
    last_pred = int(y_pred[-1])

    # Data de referência: último index do dataset tratado
    ref_date = X_test.index[-1] if hasattr(X_test, "index") else None

    # texto de confiança (apenas visual)
    conf_txt = ""
    if score01 is not None:
        if score01 >= 0.65 or score01 <= 0.35:
            conf_txt = "Confiança alta"
        elif score01 >= 0.55 or score01 <= 0.45:
            conf_txt = "Confiança média"
        else:
            conf_txt = "Confiança baixa"

    st.markdown("## Resultado")
    st.success("Tendência de **ALTA (classe 1)**." if last_pred == 1 else "Tendência de **BAIXA (classe 0)**.")

    # ---- ORIGEM (para não confundir) ----
    st.markdown("#### Origem dos dados exibidos")
    o1, o2 = st.columns(2)

    with o1:
        st.info(
            "🧠 **Previsão do MODELO (SVM)**\n\n"
            "- Fonte: **dataset tratado do notebook**\n"
            "- **Não depende do ticker** digitado\n"
            f"- Janela usada: últimos **{int(n_test)}** registros do dataset"
        )

    with o2:
        st.info(
            "📊 **Indicadores do YAHOO (informativo)**\n\n"
            f"- Ticker escolhido: **{ticker_symbol}**\n"
            "- Fonte: **Yahoo Finance**\n"
            "- **Não entra no cálculo do SVM nesta versão**"
        )

    # ---- Métricas topo (layout igual ao seu print) ----
    m1, m2, m3 = st.columns(3)

    # Data de referência: do dataset tratado (o que o modelo realmente usou)
    if isinstance(ref_date, (pd.Timestamp, datetime)):
        m1.metric("Data de referência (dataset tratado)", ref_date.date().isoformat())
    else:
        m1.metric("Data de referência (dataset tratado)", "—")

    # Último fechamento: do Yahoo do ticker (informativo)
    if yahoo_last_close is not None:
        m2.metric(
            f"Último fechamento do ticker no Yahoo ({ticker_symbol})",
            fmt_price_by_ticker(ticker_symbol, yahoo_last_close)
        )

    else:
        m2.metric(f"Último fechamento do ticker no Yahoo ({ticker_symbol})", "—")

    # Score do modelo
    if score01 is not None:
        m3.metric("Score do MODELO (0–1)", f"{score01:.3f}")
    else:
        m3.metric("Score do MODELO (0–1)", "—")

    st.caption(
        "Score do MODELO é uma transformação logística do `decision_function` "
        "(**não é probabilidade real**)."
    )
    if conf_txt:
        st.caption(conf_txt)

    # ---- Últimas features (dataset tratado) ----
    show_last_features = st.checkbox("Mostrar últimas features", value=True)
    X_last = X_test.tail(1).copy()
    if show_last_features:
        st.markdown("### Últimas features usadas (linha final do dataset tratado)")
        show_df = X_last.copy()
        show_df.insert(0, "Data", [ref_date] if ref_date is not None else ["—"])
        st.dataframe(show_df, use_container_width=True)

    # ---- Indicadores Yahoo (informativo) ----
    show_yahoo_indicators = st.checkbox("Mostrar indicadores do último dia (Yahoo)", value=True)
    if show_yahoo_indicators:
        st.markdown(f"### Indicadores do último dia (Yahoo) — {ticker_symbol} (informativo)")
        if yahoo_last_close is None or yahoo_prev_close is None:
            st.info("Yahoo não retornou dados suficientes para calcular indicadores do dia.")
        else:
            i1, i2, i3 = st.columns(3)
            i1.metric("Close (último)", fmt_price_by_ticker(ticker_symbol, yahoo_last_close))
            i2.metric("Close (anterior)", fmt_price_by_ticker(ticker_symbol, yahoo_prev_close))
            i3.metric("Variação %", f"{yahoo_pct:+.2f}%")

    # ---- Log de uso ----
    if enable_logging:
        append_log(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ticker_input": ticker,
                "ticker_symbol": ticker_symbol,
                "years_yahoo_chart": int(years),
                "n_test_dataset": int(n_test),
                "last_pred": last_pred,
                "score01": score01 if score01 is not None else "",
                "dataset_source": "local" if used_local else "url",
            }
        )
        st.success("Log salvo em logs/usage_log.csv")

    # ---- Salvar no session_state para painel de monitoramento ----
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred

    # ---- Gráfico Real x Previsto (dataset tratado) ----
    plot_df = pd.DataFrame(
        {
            "Real": y_test.values.astype(int),
            "Previsto": pd.Series(y_pred, index=y_test.index).astype(int),
        },
        index=y_test.index,
    ).reset_index().rename(columns={"index": "Data"})

    melt = plot_df.melt(id_vars="Data", value_vars=["Real", "Previsto"], var_name="Série", value_name="Classe")

    is_dt = np.issubdtype(plot_df["Data"].dtype, np.datetime64)

    chart2 = (
        alt.Chart(melt)
        .mark_line(point=True)
        .encode(
            x=alt.X("Data:T" if is_dt else "Data:O", title="Data (dataset tratado)"),
            y=alt.Y("Classe:Q", title="Classe (0=queda, 1=alta)", scale=alt.Scale(domain=[-0.1, 1.1])),
            color="Série:N",
            tooltip=["Série:N", "Classe:Q", "Data"],
        )
        .interactive()
    )
    st.altair_chart(chart2, use_container_width=True)

    st.caption("Classe 1 = alta | Classe 0 = queda")


# =========================
# Monitoramento (métricas + matriz de confusão)
# =========================
if show_monitoring:
    st.subheader("📌 Monitoramento do Modelo (métricas e performance)")
    if "y_test" not in st.session_state or "y_pred" not in st.session_state:
        st.info("Clique em **Gerar previsão** para calcular as métricas nesta janela.")
    else:
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-score", f"{f1:.3f}")

        # ---------- Matriz de confusão ----------
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])

        cm_plot = (
            cm_df.reset_index()
            .melt(id_vars="index", var_name="Pred", value_name="Count")
            .rename(columns={"index": "Real"})
        )

        # Heatmap com cor por contagem
        heatmap = (
            alt.Chart(cm_plot)
            .mark_rect()
            .encode(
                x=alt.X(
                    "Pred:N",
                    title="Previsto",
                    axis=alt.Axis(labelAngle=0),  # <-- garante horizontal
                    sort=["Pred 0", "Pred 1"],
                ),
                y=alt.Y(
                    "Real:N",
                    title="Real",
                    sort=["Real 0", "Real 1"],
                ),
                color=alt.Color("Count:Q", title="Qtde"),
                tooltip=["Real:N", "Pred:N", "Count:Q"],
            )
            .properties(height=220)
        )

        labels = (
            alt.Chart(cm_plot)
            .mark_text(fontSize=16)
            .encode(
                x=alt.X("Pred:N", sort=["Pred 0", "Pred 1"]),
                y=alt.Y("Real:N", sort=["Real 0", "Real 1"]),
                text=alt.Text("Count:Q"),
                color=alt.value("black"),
            )
            .properties(height=220)
        )

        st.write("**Matriz de confusão:**")
        col_left, col_right = st.columns([1, 1.2])
        with col_left:
            st.dataframe(cm_df, use_container_width=True)
        with col_right:
            st.altair_chart(heatmap + labels, use_container_width=True)

        # ---------- Diagnóstico (tabela + pizza) ----------
        with st.expander("Diagnóstico (distribuição Real/Previsto)"):
            real_counts = pd.Series(y_test).value_counts().sort_index()
            pred_counts = pd.Series(y_pred).value_counts().sort_index()

            # padroniza pra sempre aparecer 0 e 1
            real_counts = real_counts.reindex([0, 1], fill_value=0)
            pred_counts = pred_counts.reindex([0, 1], fill_value=0)

            diag_df = pd.DataFrame(
                {
                    "Classe": ["0 (queda)", "1 (alta)"],
                    "Real": [int(real_counts.loc[0]), int(real_counts.loc[1])],
                    "Previsto": [int(pred_counts.loc[0]), int(pred_counts.loc[1])],
                }
            )

            tcol, pcol = st.columns([1, 1.2])

            with tcol:
                st.write("**Tabela do diagnóstico**")
                st.dataframe(diag_df, use_container_width=True)

            with pcol:
                # Pizza: Real vs Previsto (lado a lado)
                pie_real = pd.DataFrame(
                    {"Classe": ["0 (queda)", "1 (alta)"], "Count": [int(real_counts.loc[0]), int(real_counts.loc[1])]}
                )
                pie_pred = pd.DataFrame(
                    {"Classe": ["0 (queda)", "1 (alta)"], "Count": [int(pred_counts.loc[0]), int(pred_counts.loc[1])]}
                )

                chart_real = (
                    alt.Chart(pie_real)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta("Count:Q"),
                        color=alt.Color("Classe:N", title=None),
                        tooltip=["Classe:N", "Count:Q"],
                    )
                    .properties(title="Distribuição REAL", height=220)
                )

                chart_pred = (
                    alt.Chart(pie_pred)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta("Count:Q"),
                        color=alt.Color("Classe:N", title=None),
                        tooltip=["Classe:N", "Count:Q"],
                    )
                    .properties(title="Distribuição PREVISTA", height=220)
                )

                st.altair_chart(chart_real | chart_pred, use_container_width=True)



# =========================
# Requisitos
# =========================
with st.expander("✅ Como isso atende o Tech Challenge"):
    st.write(
        """
- App em **Streamlit**
- Importa modelo treinado na fase 2 (**joblib**)
- **Gráfico interativo** (Altair) do Yahoo Finance para o ticker escolhido
- Painel de **métricas + matriz de confusão** (monitoramento)
- **Log de uso** em CSV (logs/usage_log.csv)
"""
    )
