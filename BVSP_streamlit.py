import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(
    page_title="PAINEL DA B3",
    layout='wide'
)

st.header("**PAINEL DE PREÇO E DIVIDENDOS DE AÇÕES DA B3**")

# Definir data de fim como ontem para evitar dados futuros
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=365 * 10)  # 10 anos de dados

# "^BVSP" parâmetro para pegar a cotação da BVSP (Ibovespa (IBOV))
ticker = st.text_input('Digite o ticker da ação', "BVSP")
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