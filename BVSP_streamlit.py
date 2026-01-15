from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pickle
# modelos
from sklearn.svm import SVC   


st.set_page_config(
    page_title="PAINEL DA BVSP",
    layout='wide'
)

st.header("**PAINEL DE PREÇO E DIVIDENDOS DE AÇÕES DA BVSP**")

# Definir data de fim como ontem para evitar dados futuros
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=365 * 10)  # 10 anos de dados



# Adicionar instruções de uso
with st.expander("📌 Instruções de Uso"):
    st.write("""
    **Tickers disponíveis:**
    - **Ibovespa:** Digite `^BVSP`
    - **Ações brasileiras:** Digite o código sem `.SA` (ex: `PETR4`, `VALE3`, `ITUB4`)
    - **ETFs:** Digite o código normalmente (ex: `BOVA11`)
    
    **Notas:**
    - Os dados são fornecidos pelo Yahoo Finance
    - Os preços estão em Reais (R$)
    - O período padrão é de 10 anos
    """)





# "^BVSP" parâmetro para pegar a cotação da BVSP (Ibovespa (IBOV))
ticker = st.text_input('Digite o ticker da ação', "^BVSP")
ticker_symbol = f"{ticker}.SA" if ticker != "^BVSP" else "^BVSP"

try:
    empresa = yf.Ticker(ticker_symbol)
    
    # Obter informações da empresa com tratamento de erro
    info = empresa.info
    
    tickerDF = empresa.history(start=start_date.strftime("%Y-%m-%d"),  
                               end=end_date.strftime("%Y-%m-%d"))
    
    # Verificar se temos dados
    if tickerDF.empty:
        st.warning(f"Não foram encontrados dados para o ticker {ticker_symbol}")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Usar get() para evitar KeyError
            nome = info.get('longName', info.get('shortName', 'Nome não disponível'))
            st.write(f"**Empresa:** {nome}")
        
        with col2:
            setor = info.get('industry', 'Setor não disponível')
            st.write(f"**Setor:** {setor}")
        
        with col3:
            # Verificar várias possíveis chaves para preço atual
            preco = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if preco:
                st.write(f"**Preço Atual:** R$ {preco:.2f}")
            else:
                st.write("**Preço Atual:** Não disponível")
        
        # Gráfico de preço de fechamento
        if not tickerDF.empty and 'Close' in tickerDF.columns:
            st.subheader("Evolução do Preço de Fechamento")
            st.line_chart(tickerDF['Close'])
        
        # Gráfico de dividendos (apenas se houver dados)
        if not tickerDF.empty and 'Dividends' in tickerDF.columns and tickerDF['Dividends'].sum() > 0:
            st.subheader("Dividendos Distribuídos")
            
            # Filtrar apenas linhas com dividendos
            dividendos_df = tickerDF[tickerDF['Dividends'] > 0]
            
            if not dividendos_df.empty:
                # Criar gráfico de barras para dividendos
                st.bar_chart(dividendos_df['Dividends'])
                
                # Mostrar tabela com dividendos recentes
                st.subheader("Últimos Dividendos")
                dividendos_recentes = dividendos_df['Dividends'].tail(10).sort_index(ascending=False)
                st.dataframe(dividendos_recentes)
            else:
                st.info("Não foram encontrados dividendos distribuídos no período.")
        else:
            st.info("Esta ação não distribuiu dividendos no período selecionado ou os dados não estão disponíveis.")
        
        # Mostrar algumas estatísticas básicas
        st.subheader("Estatísticas do Período")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            if not tickerDF.empty and 'Close' in tickerDF.columns:
                retorno_periodo = ((tickerDF['Close'].iloc[-1] / tickerDF['Close'].iloc[0]) - 1) * 100
                st.metric("Retorno no Período", f"{retorno_periodo:.2f}%")
        
        with col_stats2:
            if not tickerDF.empty and 'Volume' in tickerDF.columns:
                volume_medio = tickerDF['Volume'].mean()
                st.metric("Volume Médio Diário", f"{volume_medio:,.0f}")
        
        with col_stats3:
            if not tickerDF.empty and 'Close' in tickerDF.columns:
                preco_max = tickerDF['Close'].max()
                st.metric("Preço Máximo", f"R$ {preco_max:.2f}")
                
except Exception as e:
    st.error(f"Erro ao buscar dados: {str(e)}")
    st.info("Verifique se o ticker está correto. Para o Ibovespa use '^BVSP'. Para ações brasileiras use o código sem '.SA' (ex: PETR4, VALE3).")


dados = pd.read_csv('https://raw.githubusercontent.com/paulopetrillo/FIAP_TECH_CHALENGE_04/refs/heads/main/dados_tratados_data_correta.csv', index_col=0, parse_dates=True)

df = dados.copy()
df = df.dropna().copy()

st.info("HEADER DO DATASET")
st.write(df.head())

feature_cols = [
    # já existentes
    "Retorno","Lag1","Lag2","Lag3",
    "MM5","MM20","MM50","MM100","Volatilidade5","Volume",
    "EMA12","EMA26","MACD","MACDsig","RSI14","STOCHK","STOCHD","ATR14",
    "Ret_2d","Ret_3d","Ret_5d","Ret_10d","Ret_20d",
    "ZClose_20","ZVolume_20",

    # distâncias do preço às MMs
    "Dist_MM20_pct","Dist_MM50_pct","Dist_MM100_pct",

    # slopes
    "Slope_MM20","Slope_MM50","Slope_MM100",

    # cruzamentos
    "Cross_5_20","Cross_20_50","Cross_50_100","Cross_EMA12_26","Cross_STOCH",

    # dia da semana (versão cíclica)
    "DOW_sin","DOW_cos"
]

df = df.dropna().copy()

FEATURES = [c for c in feature_cols if c in df.columns]

# segurança: remove linhas quebradas
df_ml = df.dropna(subset=FEATURES + ["Target"]).copy()

# define X / y
X = df_ml[FEATURES].copy()
y = df_ml["Target"].astype(int).copy()

n_test = st.number_input("Número de Dias para Previsão", min_value=1, max_value=60, value=30)

# separa últimos 30 dias para TESTE
# n_test = 30
X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]


st.write("Período final após novas features:", df.index.min().date(), "->", df.index.max().date(), "| linhas:", len(df))

st.write("Treino:", X_train.index.min().date(), "->", X_train.index.max().date())
st.write("Teste :", X_test.index.min().date(),  "->", X_test.index.max().date())
st.write("Shapes:", X_train.shape, X_test.shape)

# Baseline com Random Forest (dispensa normalização)
rf = RandomForestClassifier(
    n_estimators=500,         # 500 árvores
    max_depth=8,              # profundidade máxima = 8
    min_samples_leaf=5,       # evita folhas muito pequenas
    max_features="sqrt",
    class_weight="balanced",  # balanceamento de classes
    random_state=42
)

# Treina o modelo
rf.fit(X_train, y_train)

# (função utilitária p/ alinhar colunas pela ordem vista no fit)
def align_to_fit(model, Xdf):
    cols_fit = getattr(model, "feature_names_in_", None)
    return Xdf.reindex(columns=cols_fit) if cols_fit is not None else Xdf

X_test_rf = align_to_fit(rf, X_test)
y_pred_rf = rf.predict(X_test_rf)
st.write(f"### Previsão Random Forest de Fechamento para {X_test.index[-1] + timedelta(days=1)}:")
if y_pred_rf[-1] == 0:
    st.write("Fechamento Negativo")
else:
    st.write("Fechamento Positivo") 

# Tabela de conferência: dia, abertura, fechamento, target real e previsão
resultado_rf = df.iloc[-len(y_test):][["Abertura","Fechamento"]].copy()
resultado_rf["Target_real"] = y_test.values
resultado_rf["Previsao_RF"] = y_pred_rf
resultado_rf["Acertou"] = (resultado_rf["Target_real"] == resultado_rf["Previsao_RF"]).astype(int)

st.write(resultado_rf.head(15))  # mostra os 15 primeiros

################################################################
# y_pred_svm = svm_clf_loaded.predict(X_test)

# #st.write(f"### Previsão de Fechamento para {input_data + timedelta(days=input_dias)}:")

# if y_pred_svm[-1] == 0:
#     st.write("Fechamento Negativo") 
# else:
#     st.write("Fechamento Positivo") 



# # parametros do modelo
# import pickle

# # Salva o modelo
# with open('param_name_model.pkl', 'wb') as f:
#     pickle.dump(name_model, f)

# # Carrega o modelo
# with open('param_name_model.pkl', 'rb') as f:
#     name_model_loaded = pickle.load(f)

# # 
# new_data = [[,,,]] # substitua pelos novos dados para previsão
# prediction = name_model_loaded.predict(new_data)

# if prediction == 0:
#     resultado = "Fechamento negativo"
# else:
#     resultado = "Fechamento positivo"

# print(resultado)