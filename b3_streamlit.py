import streamlit as st
import yfinance as yf 

st.set_page_config(
    page_title= "PAINEL DA B3",
    layout='wide'
)

st.header("**PAINEL DE PREÇO E DIVIDENDOS DE AÇÕES DA B3**")

ticker= st.text_input('Digite o ticker da ação', 'BBAS3')
empresa = yf.Ticker(f"{ticker}.SA")\

tickerDF = empresa.history(start = "2014-01-01",
                           end = "2025-10-30")

col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.write(f"**Empresa:**{empresa.info['longName']}")
with col2:
    st.write(f"**Setor:**{empresa.info['industry']}")
with col3:
    st.write(f"**Preço Atual:**{empresa.info['currentPrice']}")

st.line_chart(tickerDF.Close)
st.bar_chart(tickerDF.Dividends)

# http://localhost:8501/
# Local URL: http://localhost:8501
# Network URL: http://192.168.1.13:8501