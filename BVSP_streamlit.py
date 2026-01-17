from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pickle
import numpy as np

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
    - O valor indica a pontuação da ação. 
    - Não está em valores monetários.
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
        # Mostrar algumas estatísticas básicas
        st.subheader("Dados Básicos da Ação")
        col_statsA, col_statsB, col_statsC = st.columns(3)

        with col_statsA:
            # Usar get() para evitar KeyError
            nome = info.get('longName', info.get('shortName', 'Nome não disponível'))
            st.write(f"**Empresa:** \n\n {nome}")
        with col_statsB:
            st.write(f"**Ticker:** \n\n {ticker_symbol}")
        with col_statsC:
            st.write(f"**Período dos dados:** \n\n {tickerDF.index.min().date()}   até   {tickerDF.index.max().date()}")

        # Gráfico de preço de fechamento
        if not tickerDF.empty and 'Close' in tickerDF.columns:
            st.subheader("Evolução do Preço de Fechamento")
            st.line_chart(tickerDF['Close'])
                
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
                st.metric("Volume Médio Diário", f"{volume_medio:.0f}")
        
        with col_stats3:
            if not tickerDF.empty and 'Close' in tickerDF.columns:
                preco_max = tickerDF['Close'].max()
                st.metric("Pontos Máximo", f"{preco_max:.0f}")
                
except Exception as e:
    st.error(f"Erro ao buscar dados: {str(e)}")
    st.info("Verifique se o ticker está correto. Para o Ibovespa use '^BVSP'. Para ações brasileiras use o código sem '.SA' (ex: PETR4, VALE3).")


df = pd.read_csv('https://raw.githubusercontent.com/paulopetrillo/FIAP_TECH_CHALENGE_04/refs/heads/main/dados_tratados_data_correta.csv', index_col=0, parse_dates=True)

df = df.dropna().copy()

#st.info("HEADER DO DATASET")
#st.write(df.head())

# Data de previsão
st.info("Informe a data do dia de previsão (no formato YYYY-MM-DD):")
input_date = st.text_input("Data de Previsão", value=str(date.today() - timedelta(days=1))) 
try:
    input_date = pd.to_datetime(input_date)
    st.success(f"Data de previsão definida para: {input_date.date()}")
except Exception as e:
    st.error("Formato de data inválido. Use YYYY-MM-DD.")
    st.stop()

st.subheader("Entrada de Dados para Previsão")
colins1, colins2, colins3, colins4, colins5 = st.columns(5)
with colins1:
    # inserir valor de Fechamento do dia anterior à data de previsão
    st.info("Informe valor de Fechamento do dia anterior à data de previsão:")
    input_fechamento = st.number_input("Fechamento", min_value=0, step=10, format="%d")
    st.success(f"Fechamento gravado: {input_fechamento:.0f}")
with colins2:
    # inserir valor de Abertura do dia anterior à data de previsão
    st.info("Informe o valor de Abertura do dia anterior à data de previsão:")
    input_abertura = st.number_input("Abertura", min_value=0, step=10, format="%d")
    st.success(f"Abertura gravada: {input_abertura:.0f}")
with colins3:
    #inserir valor de Máxima do dia anterior à data de previsão
    st.info("Informe o valor de Máxima do dia anterior à data de previsão:")
    input_maxima = st.number_input("Máxima", min_value=0, step=10, format="%d")
    st.success(f"Máxima gravada: {input_maxima:.0f}")
with colins4:
    #inserir valor de Mínima do dia anterior à data de previsão
    st.info("Informe o valor de Mínima do dia anterior à data de previsão:")
    input_minima = st.number_input("Mínima", min_value=0, step=10, format="%d")
    st.success(f"Mínima gravada: {input_minima:.0f}")
with colins5:
    #inserir valor de Volume do dia anterior à data de previsão
    st.info("Informe o valor de Volume do dia anterior à data de previsão:")
    input_volume = st.number_input("Volume", min_value=0, step=10, format="%d")
    st.success(f"Volume gravado: {input_volume:.0f}")

# criar linha com dados informados
new_row  = pd.DataFrame({
    "Fechamento": [input_fechamento],
    "Abertura": [input_abertura],
    "Máxima": [input_maxima],
    "Mínima": [input_minima],
    "Volume": [input_volume]
}, index=[input_date])

df["Fechamento"] = new_row["Fechamento"].astype(float)
df["Abertura"] = new_row["Abertura"].astype(float)  
df["Máxima"] = new_row["Máxima"].astype(float)
df["Mínima"] = new_row["Mínima"].astype(float)   
#input_volume = df["Volume"].astype(float)

st.write("Nova linha de dados inserida:")
st.write(new_row)


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
# #################################################################
# ####### criação/cálculo das 38 features escolhidas acima ########
# #################################################################

# Retorno diário (%) {pct_change - calcula a variação percentual de uma linha em relação à anterior}
df["Retorno"] = df["Fechamento"].pct_change() * 100
input_Retorno = df["Fechamento"].pct_change() * 100

# Lags de retornos (são a forma de dar essa memória histórica para o modelo)
# shift() -> desloca a coluna para baixo em n linhas
# df["Lag1"] = df["Retorno"].shift(1)
# df["Lag2"] = df["Retorno"].shift(2)
# df["Lag3"] = df["Retorno"].shift(3)

input_Lag1 = df["Retorno"].shift(1)
input_Lag2 = df["Retorno"].shift(2)
input_Lag3 = df["Retorno"].shift(3)

# Médias móveis
# df["MM5"] = df["Fechamento"].rolling(window=5).mean()
# df["MM20"] = df["Fechamento"].rolling(window=20).mean()
# df["MM50"]  = df["Fechamento"].rolling(window=50,  min_periods=50).mean()
# df["MM100"] = df["Fechamento"].rolling(window=100, min_periods=100).mean()

input_MM5 = df["Fechamento"].rolling(window=5).mean()
input_MM20 = df["Fechamento"].rolling(window=20).mean()
input_MM50  = df["Fechamento"].rolling(window=50,  min_periods=50).mean()
input_MM100 = df["Fechamento"].rolling(window=100, min_periods=100).mean()

# Volatilidade dos últimos 5 dias (rolling std do Retorno)
#df["Volatilidade5"] = df["Retorno"].rolling(window=5).std()
input_Volatilidade5 = df["Retorno"].rolling(window=5).std()

#Volume????
input_Volume = df["Volume"]
#df["Volume"] = df["Volume"]

# EMAs e MACD (diferença de EMAs e sua média, captura momentum.)
# EMA12 = média móvel exponencial de 12 dias
#df["EMA12"] = df["Fechamento"].ewm(span=12, adjust=False).mean() 
input_EMA12 = df["Fechamento"].ewm(span=12, adjust=False).mean()

# EMA26 = média móvel exponencial de 26 dias
# df["EMA26"] = df["Fechamento"].ewm(span=26, adjust=False).mean() 
# df["MACD"]  = df["EMA12"] - df["EMA26"]
# df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()

input_EMA26 = df["Fechamento"].ewm(span=26, adjust=False).mean() 
input_MACD = input_EMA12 - input_EMA26
input_MACDsig = input_MACD.ewm(span=9, adjust=False).mean()

# RSI 14
# função auxiliar para RSI
def compute_rsi(close, window=14):
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(window).mean()
    roll_down = pd.Series(down, index=close.index).rolling(window).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi
 # usa os últimos 14 dias (padrão clássico do RSI) para calcular os ganhos e perdas médios.
#df["RSI14"] = compute_rsi(df["Fechamento"], window=14)
input_RSI14 = compute_rsi(df["Fechamento"], window=14)

# Stochastic Oscillator %K e %D (posição do fechamento dentro do range recente.)
low14  = df["Minima"].rolling(14).min()
high14 = df["Maxima"].rolling(14).max()
# df["STOCHK"] = 100 * (df["Fechamento"] - low14) / (high14 - low14 + 1e-12)
# df["STOCHD"] = df["STOCHK"].rolling(3).mean()
input_STOCHK = 100 * (df["Fechamento"] - low14) / (high14 - low14 + 1e-12)
input_STOCHD = input_STOCHK.rolling(3).mean()


# ATR proxy (True Range simples em pontos e média de 14) volatilidade em pontos.
tr1 = df["Maxima"] - df["Minima"]
tr2 = (df["Maxima"] - df["Fechamento"].shift(1)).abs()
tr3 = (df["Minima"] - df["Fechamento"].shift(1)).abs()
#df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df_TR = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#df["ATR14"] = df_TR.rolling(14).mean()
input_ATR14 = df_TR.rolling(14).mean()

# retornos em janelas (momentum) - retornos acumulados em janelas (momentum curto/médio).
for w in [2, 3, 5, 10, 20]:
    df[f"Ret_{w}d"] = df["Fechamento"].pct_change(w) * 100

input_Ret_2d = df["Fechamento"].pct_change(2) * 100
input_Ret_3d = df["Fechamento"].pct_change(3) * 100
input_Ret_5d = df["Fechamento"].pct_change(5) * 100
input_Ret_10d = df["Fechamento"].pct_change(10) * 100
input_Ret_20d = df["Fechamento"].pct_change(20) * 100

# normalizações locais (z-score rolling) para preço e volume — capturam desvios recentes - quão fora do normal” está o preço/volume em 20 dias.
    # df["ZClose_20"]  = (df["Fechamento"] - df["Fechamento"].rolling(20).mean()) / (df["Fechamento"].rolling(20).std() + 1e-12)
    # df["ZVolume_20"] = (df["Volume"]     - df["Volume"].rolling(20).mean())     / (df["Volume"].rolling(20).std() + 1e-12)
input_ZClose_20  = (df["Fechamento"] - df["Fechamento"].rolling(20).mean()) / (df["Fechamento"].rolling(20).std() + 1e-12)
input_ZVolume_20 = (df["Volume"]     - df["Volume"].rolling(20).mean())     / (df["Volume"].rolling(20).std() + 1e-12)      

# Distâncias relativas (%) do preço às MMs
# df["Dist_MM20_pct"]  = (df["Fechamento"] / df["MM20"]  - 1.0) * 100
# df["Dist_MM50_pct"]  = (df["Fechamento"] / df["MM50"]  - 1.0) * 100
# df["Dist_MM100_pct"] = (df["Fechamento"] / df["MM100"] - 1.0) * 100
input_Dist_MM20_pct  = (df["Fechamento"] / input_MM20  - 1.0) * 100
input_Dist_MM50_pct  = (df["Fechamento"] / input_MM50  - 1.0) * 100
input_Dist_MM100_pct = (df["Fechamento"] / input_MM100 - 1.0) * 100 

# "Slope" das MMs (variação % da média móvel)
for w in [20, 50, 100]:
    df[f"Slope_MM{w}"] = df[f"MM{w}"].pct_change() * 100

input_Slope_MM20 = input_MM20.pct_change() * 100
input_Slope_MM50 = input_MM50.pct_change() * 100 
input_Slope_MM100 = input_MM100.pct_change() * 100

# Cruzamentos simples
# df["Cross_5_20"]    = (df["MM5"]   > df["MM20"]).astype(int)     # se 1, tendência de curto prazo em alta.
# df["Cross_20_50"]   = (df["MM20"]  > df["MM50"]).astype(int)     # se 1, indica força no médio prazo.
# df["Cross_50_100"]  = (df["MM50"]  > df["MM100"]).astype(int)    # se 1, indica força de longo prazo.
# df["Cross_EMA12_26"]= (df["EMA12"] > df["EMA26"]).astype(int)    # se 1, esse é justamente o sinal principal do MACD → momentum positivo.
# df["Cross_STOCH"]   = (df["STOCHK"] > df["STOCHD"]).astype(int)  # se 1, sinal de compra no oscilador estocástico.
input_Cross_5_20     = (input_MM5   > input_MM20).astype(int)     # se 1, tendência de curto prazo em alta.
input_Cross_20_50    = (input_MM20  > input_MM50).astype(int)     # se 1, indica força no médio prazo.
input_Cross_50_100   = (input_MM50  > input_MM100).astype(int)    # se 1, indica força de longo prazo.
input_Cross_EMA12_26 = (input_EMA12 > input_EMA26).astype(int)    # se 1, esse é justamente o sinal principal do MACD → momentum positivo.
input_Cross_STOCH    = (input_STOCHK > input_STOCHD).astype(int)  # se 1, sinal de compra no oscilador estocástico.   


# Dia da semana (0 = segunda, 6 = domingo)
#df["DOW"] = df.index.dayofweek  # na prática teremos 0..4 (dias úteis)
input_DOW = df.index.dayofweek  # na prática teremos 0..4 (dias úteis)
# Codificação cíclica (usa 7 p/ respeitar periodicidade semanal)
# df["DOW_sin"] = np.sin(2*np.pi*df["DOW"]/7.0)
# df["DOW_cos"] = np.cos(2*np.pi*df["DOW"]/7.0)
input_DOW_sin = np.sin(2*np.pi*input_DOW/7.0)
input_DOW_cos = np.cos(2*np.pi*input_DOW/7.0)


# #################################################################
# ################## FIM calculos 38 features #####################
# #################################################################


df_38_features = df[feature_cols].copy()
# remover novos NaNs criados por indicadores
df_38_features = df_38_features.dropna().copy()

# comentaar 2 linhas abaixo
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

# # Criar inputs para os dados
# st.header("Insira os dados para previsão")

# teste de input - substituir pelos valores inseridos pelo usuário
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

Input_previsao = [[input_Retorno,input_Lag1,input_Lag2,input_Lag3,input_MM5,input_MM20,input_MM50,input_MM100,input_Volatilidade5,input_Volume,
                   input_EMA12,input_EMA26,input_MACD,input_MACDsig,input_RSI14,input_STOCHK,input_STOCHD,input_ATR14,input_Ret_2d,input_Ret_3d,
                   input_Ret_5d,input_Ret_10d,input_Ret_20d,input_ZClose_20,input_ZVolume_20,input_Dist_MM20_pct,input_Dist_MM50_pct,input_Dist_MM100_pct,input_Slope_MM20, input_Slope_MM50,
                   input_Slope_MM100,input_Cross_5_20,input_Cross_20_50,input_Cross_50_100,input_Cross_EMA12_26,input_Cross_STOCH,input_DOW_sin,input_DOW_cos
                   ]]

# st.write("Input_previsao:")
# st.write(Input_previsao)

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

#y_pred = modelo_svm_clf.predict(teste_input)
y_pred = modelo_svm_clf.predict(Input_previsao)

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