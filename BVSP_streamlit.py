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

st.info("HEADER DO DATASET")
st.write(df.head())

# Data de previsão
st.info("Informe a data do dia de previsão (no formato YYYY-MM-DD):")
input_date = st.text_input("Data de Previsão", value="2020-07-29") 
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
    st.info("Valor de Fechamento:")
    input_fechamento = df.loc[df.index.normalize() == input_date, 'Fechamento'].iloc[0]
    st.success(f"Fechamento: {input_fechamento}")
with colins2:
    # inserir valor de Abertura do dia anterior à data de previsão
    st.info("Valor de Abertura:")
    input_abertura = df.loc[df.index.normalize() == input_date, 'Abertura'].iloc[0]
    st.success(f"Abertura: {input_abertura}")
with colins3:
    #inserir valor de Máxima do dia anterior à data de previsão
    st.info("Valor de Máxima:")
    input_maxima = df.loc[df.index.normalize() == input_date, 'Maxima'].iloc[0]
    st.success(f"Máxima: {input_maxima}")
with colins4:
    #inserir valor de Mínima do dia anterior à data de previsão
    st.info("Valor de Mínima:")
    input_minima = df.loc[df.index.normalize() == input_date, 'Minima'].iloc[0]
    st.success(f"Mínima: {input_minima}")
with colins5:
    #inserir valor de Volume do dia anterior à data de previsão
    st.info("Valor de Volume:")
    input_volume = df.loc[df.index.normalize() == input_date, 'Volume'].iloc[0]
    st.success(f"Volume: {input_volume}")


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

# X_test_svm = align_to_fit(svm_clf, X_test) # garantir a mesma ordem da colunas

# Prever
y_pred_svm = modelo_svm_clf.predict(X_test)

#################################################################
st.subheader("Previsões do Modelo SVM para o Período de Teste")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# Título
st.title("📈 Avaliação do Modelo SVM")

# Matriz de Confusão
st.header("Matriz de Confusão")

# Criar figura
fig, ax = plt.subplots(figsize=(7, 5))

# Calcular matriz (substitua com seus dados reais)
mlp_cm = metrics.confusion_matrix(y_pred_svm, y_test)


col1, col2 = st.columns(2)

with col1:
        # Plotar
    sns.heatmap(mlp_cm, 
                annot=True, 
                fmt='.0f', 
                xticklabels=["Caiu", "Subiu"], 
                yticklabels=["Caiu", "Subiu"],
                cmap='Blues',
                ax=ax)

    ax.set_ylabel('Predição do Modelo')
    ax.set_xlabel('Classificação Real')
    ax.set_title('Modelo SVM')

    st.pyplot(fig)

with col2:
    # Relatório de classificação
    st.header("Relatório de Classificação")

    report = metrics.classification_report(y_test, y_pred_svm, 
                                        target_names=["Caiu", "Subiu"])
    st.text(report)



# Métricas
st.header("Métricas de Desempenho")

col1, col2, col3, col4 = st.columns(4)

with col1:
    accuracy = metrics.accuracy_score(y_test, y_pred_svm)
    st.metric("Acurácia", f"{accuracy:.2%}")

with col2:
    precision = metrics.precision_score(y_test, y_pred_svm, average='binary')
    st.metric("Precisão", f"{precision:.2%}")

with col3:
    recall = metrics.recall_score(y_test, y_pred_svm, average='binary')
    st.metric("Recall", f"{recall:.2%}")

with col4:
    f1 = metrics.f1_score(y_test, y_pred_svm, average='binary')
    st.metric("F1-Score", f"{f1:.2%}")


#################################################################
# # Criar inputs para os dados
# st.header("Insira os dados para previsão")

# # teste de input - substituir pelos valores inseridos pelo usuário
# teste_input=[[-0.0796797263681647,-1.9969716881410136,-0.5643672174612924,
#       1.4369554985640187,104.093,102.18675,96.97518,86.91625,
#       1.2265044439895267,10900000,103.14725011259122,101.33530451543,
#       1.8119455971612268,1.952086539617413,57.75068102212942,
#       48.16017316016453,67.04462503854658,1.8918571428571431,
#       -2.075060232932413,-2.6277164906964634,-1.5764235190520504,
#       -1.5283550073736496,3.9348272132771367,0.2785754881872977,
#       0.976829878934459,0.6294847423956806,6.037441745403305,
#       18.30929199085325,0.1908481147069096,0.427849287359705,
#       0.1222899319652581,1,1,1,1,0,0,1]]

# Input_previsao = [[input_Retorno,input_Lag1,input_Lag2,input_Lag3,input_MM5,input_MM20,input_MM50,input_MM100,input_Volatilidade5,input_Volume,
#                    input_EMA12,input_EMA26,input_MACD,input_MACDsig,input_RSI14,input_STOCHK,input_STOCHD,input_ATR14,input_Ret_2d,input_Ret_3d,
#                    input_Ret_5d,input_Ret_10d,input_Ret_20d,input_ZClose_20,input_ZVolume_20,input_Dist_MM20_pct,input_Dist_MM50_pct,input_Dist_MM100_pct,input_Slope_MM20, input_Slope_MM50,
#                    input_Slope_MM100,input_Cross_5_20,input_Cross_20_50,input_Cross_50_100,input_Cross_EMA12_26,input_Cross_STOCH,input_DOW_sin,input_DOW_cos
#                    ]]

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
#y_pred = modelo_svm_clf.predict(Input_previsao)

# # st.write(f"### Previsão SVM de Fechamento para o próximo dia:")
# # if y_pred == 0:
# #     st.write("Fechamento Negativo")
# # else:
# #     st.write("Fechamento Positivo") 

# # st.write(y_pred)

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