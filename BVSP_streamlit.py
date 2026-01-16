from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pickle

# modelos
from sklearn.svm import SVC   
from datetime import date


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


df = pd.read_csv('https://raw.githubusercontent.com/paulopetrillo/FIAP_TECH_CHALENGE_04/refs/heads/main/dados_tratados_data_correta.csv', index_col=0, parse_dates=True)

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

df_38_features = df[feature_cols].copy()
df_38_features = df_38_features.dropna().copy()
st.info("DATASET COM 38 FEATURES")
st.write(df_38_features.head())

# # segurança: remove linhas quebradas
# df_ml = df.dropna(subset=FEATURES + ["Target"]).copy()

# define X / y
X = df_38_features.copy()
y = df["Target"].astype(int).copy()

n_test = st.number_input("Número de Dias para Previsão", min_value=1, max_value=60, value=30)

# separa últimos 30 dias para TESTE
# n_test = 30
X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]


st.write("Período final após novas features:", df.index.min().date(), "->", df.index.max().date(), "| linhas:", len(df))

st.write("Treino:", X_train.index.min().date(), "->", X_train.index.max().date())
st.write("Teste :", X_test.index.min().date(),  "->", X_test.index.max().date())
st.write("Shapes:", X_train.shape, X_test.shape)

# Carregar o arquivo pickle
with open('svm_clf.pkl', 'rb') as arquivo:
    modelo_svm_clf = pickle.load(arquivo)

# Criar inputs para os dados
st.header("Insira os dados para previsão")

# Exemplo para um modelo com 3 features
teste_input=[[-0.0796797263681647,-1.9969716881410136,-0.5643672174612924,
      1.4369554985640187,104.093,102.18675,96.97518,86.91625,
      1.2265044439895267,10900000,103.14725011259122,101.33530451543,
      1.8119455971612268,1.952086539617413,57.75068102212942,
      48.16017316016453,67.04462503854658,1.8918571428571431,
      -2.075060232932413,-2.6277164906964634,-1.5764235190520504,
      -1.5283550073736496,3.9348272132771367,0.2785754881872977,
      0.976829878934459,0.6294847423956806,6.037441745403305,
      18.30929199085325,0.1908481147069096,0.427849287359705,
      0.1222899319652581,1,1,1,1,0,0,1]]

st.write("Input de teste:")
st.write(teste_input)

# Depois calcular os 38 valores.
# feature1 = st.number_input("Feature 1", value=0.0)
# feature2 = st.number_input("Feature 2", value=0.0)
# feature3 = st.number_input("Feature 3", value=0.0)
# ...
# input_data = [[feature1, feature2, feature3]]

# # Botão para fazer previsão
# if st.button("Prever"):
#     # Criar array com os dados de entrada
#     dados_entrada = np.array([[feature1, feature2, feature3]])
    
#     # Fazer previsão
#     previsao = modelo.predict(dados_entrada)
    
#     # Mostrar resultado
#     st.success(f"Previsão: {previsao[0]}")

y_pred = modelo_svm_clf.predict(teste_input)

st.write(f"### Previsão SVM de Fechamento para o próximo dia:")
if y_pred == 0:
    st.write("Fechamento Negativo")
else:
    st.write("Fechamento Positivo") 

st.write(y_pred)

#################################################################
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