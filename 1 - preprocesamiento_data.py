import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta



############################################
# 1. Carga y unificación de variables
############################################

# Cargar datos macroeconómicos
treasury_df = pd.read_csv('Treasury_10Y.csv')
treasury_df['Date'] = pd.to_datetime(treasury_df['Unnamed: 0'])
treasury_df.set_index('Date', inplace=True)
treasury_df = treasury_df[['Treasury_10Y']]

vix_df = pd.read_csv('VIX.csv')
vix_df['Date'] = pd.to_datetime(vix_df['observation_date'])
vix_df.set_index('Date', inplace=True)
vix_df = vix_df[['VIXCLS']]

dxy_df = pd.read_csv('DXY.csv')
dxy_df['Date'] = pd.to_datetime(dxy_df['observation_date'])
dxy_df.set_index('Date', inplace=True)
dxy_df = dxy_df[['DTWEXBGS']]

# Crear DataFrame macro unificado con calendario de días hábiles
macro_dates = pd.date_range(start='2020-01-02', end='2025-02-28', freq='B')
macro_data = pd.DataFrame(index=macro_dates)
macro_data['Treasury_10Y'] = treasury_df['Treasury_10Y'].reindex(macro_dates).ffill()
macro_data['VIX'] = vix_df['VIXCLS'].reindex(macro_dates).ffill()
macro_data['DXY'] = dxy_df['DTWEXBGS'].reindex(macro_dates).ffill()

# Cargar datos de acciones (sin indicadores técnicos iniciales)
symbols = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'META']
stock_data = {}
for symbol in symbols:
    df = pd.read_csv(f"{symbol}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Se conservan únicamente las columnas básicas
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_data[symbol] = df

# Determinar fechas comunes entre todas las acciones y datos macro
common_dates = sorted(list(set.intersection(*[set(df.index) for df in stock_data.values()],
                                            set(macro_data.index))))

# 1. Obtener el conjunto de todas las fechas presentes en las 5 acciones (left join)
all_stock_dates = sorted(list(set.union(*[set(df.index) for df in stock_data.values()])))

# 2. Reindexar cada DataFrame de acción usando all_stock_dates
aligned_stocks = {}
for symbol, df in stock_data.items():
    # Se reindexa usando el conjunto de fechas de las acciones y se rellenan los valores faltantes
    aligned_stocks[symbol] = df.reindex(all_stock_dates).ffill().bfill()

# 3. "Adjuntar" las variables macro a estas fechas (left join)
aligned_macro = macro_data.reindex(all_stock_dates).ffill().bfill()

############################################
# 2. Cálculo de indicadores técnicos y escalado
############################################

def calculate_state_features(df):
    """
    Calcula 10 características para cada acción:
      - 5 precios básicos: Open, High, Low, Close, Volume
      - 2 medias móviles: SMA_20 y SMA_50
      - 2 indicadores de momentum: RSI y ROC
      - 1 indicador de volatilidad: Volatility (desviación estándar de los retornos)
    """
    df_features = pd.DataFrame(index=df.index)
    
    # Precios básicos (5 features)
    df_features['Open'] = df['Open']
    df_features['High'] = df['High']
    df_features['Low'] = df['Low']
    df_features['Close'] = df['Close']
    df_features['Volume'] = df['Volume']
    
    # Tendencia (7 features)
    df_features['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df_features['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df_features['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df_features['MACD'] = ta.trend.macd_diff(df['Close'])
    df_features['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
    df_features['Relative_Price_20'] = (df['Close'] / df_features['SMA_20']) - 1
    df_features['Relative_Price_50'] = (df['Close'] / df_features['SMA_50']) - 1
    
    # Momentum (3 features)
    df_features['RSI'] = ta.momentum.rsi(df['Close'])
    df_features['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df_features['ROC'] = ta.momentum.roc(df['Close'])
    
    # Volatilidad (4 features)
    df_features['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
    df_features['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
    df_features['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df_features['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # Volumen
    df_features['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df_features['ADI'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Retornos
    df_features['Returns'] = df['Close'].pct_change()
    df_features['Returns_5d'] = df['Close'].pct_change(periods=5)
    df_features['Returns_20d'] = df['Close'].pct_change(periods=20)
    
    return df_features

# Calcular indicadores técnicos y escalar datos para cada acción
final_stock_data = {}
stock_scaler = MinMaxScaler()

for symbol, df in aligned_stocks.items():

    # Calcular indicadores técnicos a partir de los datos básicos
    features_df = calculate_state_features(df)
    # Rellenar valores faltantes
    features_df = features_df.ffill().bfill()   
    # Escalar las características
    scaled_values = stock_scaler.fit_transform(features_df)
    scaled_df = pd.DataFrame(scaled_values, index=features_df.index, columns=features_df.columns)
    final_stock_data[symbol] = scaled_df
    # Guardar el resultado en archivo (opcional)
    scaled_df.to_csv(f"{symbol}_final.csv")

# Escalar datos macro
macro_scaler = MinMaxScaler()
scaled_macro = pd.DataFrame(macro_scaler.fit_transform(aligned_macro), 
                            index=aligned_macro.index, 
                            columns=aligned_macro.columns)
scaled_macro.index.name = 'Date'
scaled_macro.to_csv("macro_final.csv")

print("Proceso completado: datos unificados, alineados, indicadores técnicos calculados y escalados.")
