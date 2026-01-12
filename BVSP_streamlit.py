import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pickle
# modelos
from sklearn.svm import SVC                                        # SVM


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


dados = pd.read_csv('https://raw.githubusercontent.com/paulopetrillo/FIAP_TECH_CHALENGE_04/refs/heads/main/dados_tratados.csv')

# st.info("HEADER DO DATASET")
# st.write(dados.head())

# st.info("Resumo do Dataset")
# st.write(dados.describe())

# st.info("Informações do Dataset")
# st.write(dados.info())

# # Criar o DataFrame
# data_table = {
#     'index': list(range(38)),
#     'feature': [
#         'Retorno', 'Ret_3d', 'Lag3', 'RSI14', 'Dist_MM20_pct', 'Cross_STOCH',
#         'DOW_cos', 'Cross_50_100', 'MM5', 'MM20', 'MACDsig', 'Ret_2d', 'MM50',
#         'MACD', 'Ret_5d', 'ATR14', 'Cross_20_50', 'Cross_5_20', 'Dist_MM100_pct',
#         'Cross_EMA12_26', 'EMA26', 'EMA12', 'Slope_MM100', 'Slope_MM20', 'MM100',
#         'Volume', 'Dist_MM50_pct', 'STOCHD', 'Slope_MM50', 'Ret_20d', 'Ret_10d',
#         'ZClose_20', 'STOCHK', 'Lag2', 'Volatilidade5', 'ZVolume_20', 'DOW_sin',
#         'Lag1'
#     ],
#     'importancia_media': [
#         0.100000, 0.018333, 0.015000, 0.005000, 0.005000, 0.003333,
#         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
#         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
#         0.000000, 0.000000, 0.000000, -0.001667, -0.001667, -0.005000,
#         -0.005000, -0.005000, -0.006667, -0.008333, -0.010000, -0.011667,
#         -0.015000, -0.015000, -0.018333, -0.026667, -0.043333, -0.061667
#     ],
#     'importancia_std': [
#         0.050553, 0.022298, 0.026822, 0.030322, 0.028431, 0.027689,
#         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.027889, 0.000000,
#         0.000000, 0.014907, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
#         0.000000, 0.000000, 0.000000, 0.007265, 0.007265, 0.011902,
#         0.015899, 0.028431, 0.024944, 0.014434, 0.026034, 0.028431,
#         0.024664, 0.019650, 0.016583, 0.022608, 0.028087, 0.026405
#     ]
# }

# df_table = pd.DataFrame(data_table)

# # Título da tabela
# st.header("Importância das Variáveis por Permutação")

# # Exibir a tabela no Streamlit
# st.dataframe(df_table, hide_index=True, use_container_width=True)

st.write('### Insira novos valores para previsão de fechamento da ação:')
input_data = st.date_input("Data da Previsão")
input_open = st.number_input("Preço de Abertura", format="%.3f")
input_high = st.number_input("Preço Máximo", format="%.3f")
input_low = st.number_input("Preço Mínimo", format="%.3f")
input_close = st.number_input("Preço de Fechamento", format="%.3f")

# calcular outros parâmetros necessários para o modelo

# Carrega o modelo
with open('svm_clf.pkl', 'rb') as f:
    svm_clf_loaded = pickle.load(f)

y_pred_svm = svm_clf_loaded.predict(X_test_svm)


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