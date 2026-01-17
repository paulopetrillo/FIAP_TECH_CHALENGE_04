from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# modelos
from datetime import date
from sklearn.svm import SVC   
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
                st.metric("Volume Médio Diário", f"{volume_medio:,.0f}")
        
        with col_stats3:
            if not tickerDF.empty and 'Close' in tickerDF.columns:
                preco_max = tickerDF['Close'].max()
                st.metric("Pontos Máximo", f"{preco_max:,.0f}")
                
except Exception as e:
    st.error(f"Erro ao buscar dados: {str(e)}")
    st.info("Verifique se o ticker está correto. Para o Ibovespa use '^BVSP'. Para ações brasileiras use o código sem '.SA' (ex: PETR4, VALE3).")

# Carregar dados
df = pd.read_csv('https://raw.githubusercontent.com/paulopetrillo/FIAP_TECH_CHALENGE_04/refs/heads/main/dados_tratados_data_correta.csv', index_col=0, parse_dates=True)
df = df.dropna().copy()

st.info("HEADER DO DATASET")
st.write(df.head())

# Data de previsão
st.info("Informe a data do dia de previsão (no formato YYYY-MM-DD):")
input_date_str = st.text_input("Data de Previsão", value="2020-07-29") 

try:
    input_date = pd.to_datetime(input_date_str)
    st.success(f"Data de previsão definida para: {input_date.date()}")
    
    # Verificar se a data existe no DataFrame
    # if input_date.date() not in df.index.normalize():
    #     st.error(f"A data {input_date.date()} não foi encontrada no dataset.")
    #     st.stop()
        
    input_date = '2020-07-29'
    # Obter valores para a data específica
    input_fechamento = df.loc[df.index.normalize() == input_date, 'Fechamento'].iloc[0]
    input_abertura = df.loc[df.index.normalize() == input_date, 'Abertura'].iloc[0]
    input_maxima = df.loc[df.index.normalize() == input_date, 'Maxima'].iloc[0]
    input_minima = df.loc[df.index.normalize() == input_date, 'Minima'].iloc[0]
    input_volume = df.loc[df.index.normalize() == input_date, 'Volume'].iloc[0]
    
    st.subheader("Entrada de Dados para Previsão")
    colins1, colins2, colins3, colins4, colins5 = st.columns(5)
    
    with colins1:
        st.info("Valor de Fechamento:")
        st.success(f"Fechamento: {input_fechamento:,.2f}")
    
    with colins2:
        st.info("Valor de Abertura:")
        st.success(f"Abertura: {input_abertura:,.2f}")
    
    with colins3:
        st.info("Valor de Máxima:")
        st.success(f"Máxima: {input_maxima:,.2f}")
    
    with colins4:
        st.info("Valor de Mínima:")
        st.success(f"Mínima: {input_minima:,.2f}")
    
    with colins5:
        st.info("Valor de Volume:")
        st.success(f"Volume: {input_volume:,.0f}")
        
except Exception as e:
    st.error(f"Erro ao processar data: {str(e)}")
    st.stop()

###################################################################
######################## Previsão Futura ##########################

class SVMPredictor:
    """
    Classe para fazer previsões usando modelo SVM pré-treinado.
    """
    
    def __init__(self, model_path='svm_clf.pkl'):
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            st.success(f"Modelo carregado com sucesso de {model_path}")
        except FileNotFoundError:
            st.error(f"Arquivo {model_path} não encontrado!")
            raise
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {str(e)}")
            raise
        
        self.scaler = StandardScaler()
        self.feature_names = [
            "Retorno", "Lag1", "Lag2", "Lag3",
            "MM5", "MM20", "MM50", "MM100", "Volatilidade5", "Volume",
            "EMA12", "EMA26", "MACD", "MACDsig", "RSI14", "STOCHK", "STOCHD", "ATR14",
            "Ret_2d", "Ret_3d", "Ret_5d", "Ret_10d", "Ret_20d",
            "ZClose_20", "ZVolume_20",
            "Dist_MM20_pct", "Dist_MM50_pct", "Dist_MM100_pct",
            "Slope_MM20", "Slope_MM50", "Slope_MM100",
            "Cross_5_20", "Cross_20_50", "Cross_50_100", "Cross_EMA12_26", "Cross_STOCH",
            "DOW_sin", "DOW_cos"
        ]
    
    def calculate_all_features(self, df):
        """
        Calcula todas as features necessárias para o modelo.
        """
        df = df.copy()
        
        # 1. Retorno diário
        df["Retorno"] = df["Fechamento"].pct_change() * 100
        
        # 2. Lags
        df["Lag1"] = df["Retorno"].shift(1)
        df["Lag2"] = df["Retorno"].shift(2)
        df["Lag3"] = df["Retorno"].shift(3)
        
        # 3. Médias Móveis
        df["MM5"] = df["Fechamento"].rolling(window=5).mean()
        df["MM20"] = df["Fechamento"].rolling(window=20).mean()
        df["MM50"] = df["Fechamento"].rolling(window=50).mean()
        df["MM100"] = df["Fechamento"].rolling(window=100).mean()
        
        # 4. Volatilidade
        df["Volatilidade5"] = df["Retorno"].rolling(window=5).std()
        
        # 5. Volume já existe
        
        # 6. EMA e MACD
        df["EMA12"] = df["Fechamento"].ewm(span=12, adjust=False).mean()
        df["EMA26"] = df["Fechamento"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        # 7. RSI
        def compute_rsi(series, window=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df["RSI14"] = compute_rsi(df["Fechamento"], window=14)
        
        # 8. Stochastic
        low_14 = df["Minima"].rolling(window=14).min()
        high_14 = df["Maxima"].rolling(window=14).max()
        df["STOCHK"] = 100 * (df["Fechamento"] - low_14) / (high_14 - low_14 + 1e-10)
        df["STOCHD"] = df["STOCHK"].rolling(window=3).mean()
        
        # 9. ATR
        high_low = df["Maxima"] - df["Minima"]
        high_close = (df["Maxima"] - df["Fechamento"].shift()).abs()
        low_close = (df["Minima"] - df["Fechamento"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR14"] = tr.rolling(window=14).mean()
        
        # 10. Retornos em janelas diferentes
        for window in [2, 3, 5, 10, 20]:
            df[f"Ret_{window}d"] = df["Fechamento"].pct_change(window) * 100
        
        # 11. Z-Scores
        df["ZClose_20"] = (df["Fechamento"] - df["Fechamento"].rolling(20).mean()) / (df["Fechamento"].rolling(20).std() + 1e-10)
        df["ZVolume_20"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-10)
        
        # 12. Distâncias às MMs
        df["Dist_MM20_pct"] = (df["Fechamento"] / df["MM20"] - 1) * 100
        df["Dist_MM50_pct"] = (df["Fechamento"] / df["MM50"] - 1) * 100
        df["Dist_MM100_pct"] = (df["Fechamento"] / df["MM100"] - 1) * 100
        
        # 13. Slope das MMs
        df["Slope_MM20"] = df["MM20"].pct_change() * 100
        df["Slope_MM50"] = df["MM50"].pct_change() * 100
        df["Slope_MM100"] = df["MM100"].pct_change() * 100
        
        # 14. Cruzamentos
        df["Cross_5_20"] = (df["MM5"] > df["MM20"]).astype(int)
        df["Cross_20_50"] = (df["MM20"] > df["MM50"]).astype(int)
        df["Cross_50_100"] = (df["MM50"] > df["MM100"]).astype(int)
        df["Cross_EMA12_26"] = (df["EMA12"] > df["EMA26"]).astype(int)
        df["Cross_STOCH"] = (df["STOCHK"] > df["STOCHD"]).astype(int)
        
        # 15. Dia da semana
        df["DOW"] = df.index.dayofweek
        df["DOW_sin"] = np.sin(2 * np.pi * df["DOW"] / 7)
        df["DOW_cos"] = np.cos(2 * np.pi * df["DOW"] / 7)
        
        return df
    
    def predict_for_date(self, full_df, target_date):
        """
        Faz previsão para uma data específica usando dados históricos.
        """
        # Encontra o índice da data alvo
        target_idx = full_df.index.get_loc(target_date)
        
        # Precisamos de dados suficientes para calcular todas as features
        min_required = max(100, target_idx)  # Pelo menos 100 dias para MMs
        
        if target_idx < 100:
            st.warning(f"Para prever para {target_date}, precisamos de pelo menos 100 dias de dados históricos.")
            return None
        
        # Pega dados até a data alvo (inclusive)
        data_until_target = full_df.iloc[:target_idx + 1].copy()
        
        # Calcula features
        data_with_features = self.calculate_all_features(data_until_target)
        
        # Pega a última linha (data alvo)
        last_row = data_with_features.iloc[[-1]]
        
        # Seleciona apenas as features necessárias
        features = last_row[self.feature_names]
        
        # Faz a previsão
        prediction = self.model.predict(features)
        
        # Tenta obter probabilidades
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
        
        return {
            'prediction': prediction[0],
            'probabilities': probabilities[0] if probabilities is not None else None,
            'features': last_row.iloc[0].to_dict()
        }

# Instanciar o predictor
try:
    predictor = SVMPredictor('svm_clf.pkl')
    
    # Fazer previsão para a data especificada
    st.subheader("Previsão para a Data Específica")
    
    result = predictor.predict_for_date(df, input_date)
    
    if result:
        prediction = result['prediction']
        probabilities = result['probabilities']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 0:
                st.error("📉 **Previsão: Fechamento Negativo**")
            else:
                st.success("📈 **Previsão: Fechamento Positivo**")
        
        with col2:
            if probabilities is not None:
                st.metric("Probabilidade de Subir", f"{probabilities[1]:.2%}")
        
        with col3:
            if probabilities is not None:
                st.metric("Probabilidade de Cair", f"{probabilities[0]:.2%}")
        
        # Mostrar algumas features importantes
        st.subheader("Algumas Features Calculadas")
        features_to_show = ['RSI14', 'MACD', 'STOCHK', 'Volatilidade5', 'Retorno']
        features_df = pd.DataFrame({
            'Feature': features_to_show,
            'Valor': [result['features'].get(f, 'N/A') for f in features_to_show]
        })
        st.table(features_df)
        
except Exception as e:
    st.error(f"Erro ao fazer previsão: {str(e)}")

####################### Avaliação do Modelo #######################
st.divider()
st.subheader("📊 Avaliação do Modelo SVM")

# Calcular features para todo o dataset
df_with_features = predictor.calculate_all_features(df)

# Selecionar features
feature_cols = predictor.feature_names
df_38_features = df_with_features[feature_cols].copy()
df_38_features = df_38_features.dropna().copy()

# Garantir que temos target correspondente
# Remover linhas onde não temos target
available_idx = df_38_features.index
df_target = df.loc[available_idx, 'Target'].astype(int)

# Definir tamanho do teste
n_test = st.number_input("Número de Dias para Previsão (Teste)", 
                         min_value=1, max_value=60, value=30)

# Separar treino e teste
if len(df_38_features) > n_test:
    X_train = df_38_features.iloc[:-n_test]
    X_test = df_38_features.iloc[-n_test:]
    y_train = df_target.iloc[:-n_test]
    y_test = df_target.iloc[-n_test:]
    
    # Carregar modelo SVM
    with open('svm_clf.pkl', 'rb') as arquivo:
        modelo_svm_clf = pickle.load(arquivo)
    
    # Ajustar o scaler com dados de treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fazer previsões
    y_pred_svm = modelo_svm_clf.predict(X_test_scaled)
    
    # Métricas de desempenho
    st.subheader("Métricas de Desempenho")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = metrics.accuracy_score(y_test, y_pred_svm)
        st.metric("Acurácia", f"{accuracy:.2%}")
    
    with col2:
        precision = metrics.precision_score(y_test, y_pred_svm, average='binary', zero_division=0)
        st.metric("Precisão", f"{precision:.2%}")
    
    with col3:
        recall = metrics.recall_score(y_test, y_pred_svm, average='binary', zero_division=0)
        st.metric("Recall", f"{recall:.2%}")
    
    with col4:
        f1 = metrics.f1_score(y_test, y_pred_svm, average='binary', zero_division=0)
        st.metric("F1-Score", f"{f1:.2%}")
    
    # Matriz de Confusão
    st.subheader("Matriz de Confusão")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = metrics.confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Caiu", "Subiu"], 
                yticklabels=["Caiu", "Subiu"],
                ax=ax)
    ax.set_xlabel('Predição')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusão - SVM')
    st.pyplot(fig)
    
    # Relatório de Classificação
    st.subheader("Relatório de Classificação")
    report = metrics.classification_report(y_test, y_pred_svm, 
                                          target_names=["Caiu", "Subiu"],
                                          output_dict=False)
    st.text(report)
    
    # Previsões vs Real
    st.subheader("Previsões vs Real (Últimos 20 dias)")
    results_df = pd.DataFrame({
        'Data': X_test.index[-20:],
        'Real': y_test[-20:].values,
        'Previsto': y_pred_svm[-20:]
    })
    results_df['Real'] = results_df['Real'].map({0: 'Caiu', 1: 'Subiu'})
    results_df['Previsto'] = results_df['Previsto'].map({0: 'Caiu', 1: 'Subiu'})
    results_df['Acerto'] = results_df['Real'] == results_df['Previsto']
    
    st.dataframe(results_df)
    
    # Estatísticas de acerto
    acertos = results_df['Acerto'].sum()
    total = len(results_df)
    st.metric("Taxa de Acerto (últimos 20 dias)", f"{acertos/total:.2%}")
    
else:
    st.warning("Dataset muito pequeno para separação treino-teste. Adicione mais dados.")

# Previsão para o próximo dia
st.divider()
st.subheader("🔮 Previsão para o Próximo Dia Útil")

if len(df) > 0:
    # Supondo que queremos prever o dia seguinte à última data disponível
    last_date = df.index[-1]
    next_date = last_date + pd.Timedelta(days=1)
    
    # Para prever o próximo dia, precisaríamos dos dados do dia atual
    # Como exemplo, vamos usar os últimos dados disponíveis
    last_row = df.iloc[[-1]]
    
    # Criar um DataFrame simulado para o próximo dia (usando último dado disponível)
    next_day_data = last_row.copy()
    next_day_data.index = [next_date]
    
    # Fazer previsão
    try:
        next_prediction = predictor.predict_for_date(df, last_date)
        if next_prediction:
            st.write(f"**Última data disponível:** {last_date.date()}")
            if next_prediction['prediction'] == 0:
                st.error(f"📉 **Previsão para próximo dia: Fechamento Negativo**")
            else:
                st.success(f"📈 **Previsão para próximo dia: Fechamento Positivo**")
            
            if next_prediction['probabilities'] is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prob. de Subir", f"{next_prediction['probabilities'][1]:.2%}")
                with col2:
                    st.metric("Prob. de Cair", f"{next_prediction['probabilities'][0]:.2%}")
    except Exception as e:
        st.warning(f"Não foi possível fazer previsão para próximo dia: {str(e)}")