import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime.lime_tabular
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
from transformers import pipeline
import os
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("Stock Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("Stock Selection")
    
    # Chỉ giữ lại single ticker
    ticker = st.selectbox("Select a Ticker:", [
        "Apple (AAPL)", "Amazon (AMZN)", "Google (GOOGL)", "NVIDIA (NVDA)",
        "Microsoft (MSFT)", "Meta (META)", "Intel (INTC)", "Qualcomm (QCOM)", "Tesla (TSLA)"
    ])
    ticker_symbol = ticker.split()[1].strip("()")  # Lấy mã ticker (ví dụ: AAPL)

    st.write("START DATE")
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    st.write("END DATE")
    end_date = st.date_input("End Date", value=datetime.now())

    date_diff = (end_date - start_date).days
    if date_diff < 50:
        st.error(f"The selected date range is {date_diff} days, which is too short. Please select a date range of at least 50 days to compute indicators like SMA50.")
        st.stop()

    data_freq = "1d"
    chart_type = st.selectbox("Chart Type", ["Candle", "Line"], index=0)
    st.write("TECHNICAL INDICATORS OPTIONS")
    tech_indicators = st.selectbox("Choose Technical Indicators:", [
        "None", "SMA20", "SMA50", "EMA20", "EMA50", "RSI", "MACD", "Bollinger Bands"
    ], index=0)

# Attention Layer cho LSTM
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e, axis=-1)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context, alpha

# Hàm lấy dữ liệu từ Yahoo Finance
def load_data_yfinance(symbol, interval='1d', start_date=None, end_date=None):
    try:
        today = datetime.now().date()
        if start_date > today or end_date > today:
            st.error("Start or end date cannot be in the future!")
            return None
        
        if start_date >= end_date:
            st.error("Start date must be earlier than end date!")
            return None

        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date + timedelta(days=1), interval=interval)
        
        if data.empty:
            st.error(f"No data available for {symbol} in this date range! Please check the ticker or date range.")
            return None
        
        if 'Close' not in data.columns:
            st.error(f"Data for {symbol} does not contain 'Close' column. Please check the data source.")
            return None
        
        if data['Close'].isna().all():
            st.error(f"All 'Close' prices for {symbol} are NaN in this date range. Please select a different date range or ticker.")
            return None
        
        data = data.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        data['Adj Close'] = data['Close']
        
        data['% Change'] = data['Close'].pct_change() * 100
        
        data['RSI'] = compute_rsi(data['Close'])
        data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])
        data['Bollinger_Upper'], data['Bollinger_Lower'] = compute_bollinger_bands(data['Close'])
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        data = data.dropna()
        
        if data.empty:
            st.error(f"No valid data after processing for {symbol}. Please ensure the data contains valid price information.")
            return None
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
        return None

# Hàm tính RSI
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Hàm tính MACD
def compute_macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Hàm tính Bollinger Bands
def compute_bollinger_bands(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Hàm lấy tin tức từ NewsAPI
def fetch_news(ticker_symbol):
    ticker_to_company = {
        "AAPL": "Apple", "AMZN": "Amazon", "GOOGL": "Google", "NVDA": "NVIDIA",
        "MSFT": "Microsoft", "META": "Meta", "INTC": "Intel", "QCOM": "Qualcomm", "TSLA": "Tesla"
    }
    company_name = ticker_to_company.get(ticker_symbol, ticker_symbol)
    query = f"{ticker_symbol} OR {company_name}"
    NEWS_API_KEY = "a29bdff41651403a9351130aa40fcd20"
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=10"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        if news_data.get("status") != "ok":
            st.error("Error fetching news: " + news_data.get("message", "Unknown error"))
            return []
        articles = news_data.get("articles", [])
        if not articles:
            st.warning(f"No news found for {ticker_symbol} or {company_name}.")
        return articles
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

# Hàm phân tích cảm xúc với transformers
def analyze_sentiment(text):
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
        label = result['label']
        score = result['score']
        if label == "POSITIVE":
            sentiment = "Positive"
        elif label == "NEGATIVE":
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return sentiment, score
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return "Unknown", 0.0

# Hàm tính các chỉ số đánh giá
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    return mae, mse, rmse, r2, mape, smape

# Lấy dữ liệu cho ticker đã chọn
df = load_data_yfinance(ticker_symbol, interval=data_freq, start_date=start_date, end_date=end_date)

# Main Panel
if df is not None and not df.empty:
    tab1, tab2, tab3, tab4 = st.tabs(["Pricing Data", "News", "Sentiment Analysis", "Stock Price Prediction"])

    with tab1:
        st.success(f"Data for {ticker_symbol} loaded successfully!")
        
        if df['Close'].isna().all():
            st.error("No valid price data to compute metrics. Please check the data.")
            annual_return = std_dev = risk_adj_return = "N/A"
        else:
            returns = df['Close'].pct_change().dropna()
            if returns.empty:
                st.warning("Not enough data to compute metrics (requires at least 2 days of data).")
                annual_return = std_dev = risk_adj_return = "N/A"
            else:
                annual_factor = 252
                annual_return = returns.mean() * annual_factor * 100
                std_dev = returns.std() * np.sqrt(annual_factor) * 100
                risk_adj_return = annual_return / std_dev if std_dev != 0 else 0
                annual_return = f"{annual_return:.2f}%"
                std_dev = f"{std_dev:.2f}%"
                risk_adj_return = f"{risk_adj_return:.2f}"

        st.write(f"Showing data with '{data_freq}' frequency:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Annual Return", annual_return)
        col2.metric("Standard Deviation", std_dev)
        col3.metric("Risk Adjusted Return", risk_adj_return)

        st.write("### Pricing Data")
        display_df = df[['Close', 'High', 'Low', 'Open', 'Volume', '% Change']].reset_index()
        display_df = display_df.rename(columns={'index': 'Date'})
        display_df['% Change'] = display_df['% Change'].apply(lambda x: f"{x:.3f}%" if not pd.isna(x) else "N/A")
        st.dataframe(display_df)

        st.write("### Data Visualization")
        
        st.write("#### Price Chart")
        if df['Close'].isna().all():
            st.error("No valid price data to plot the chart. Please check the data.")
        else:
            fig_price = go.Figure()
            if chart_type == "Candle":
                fig_price.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Candlestick'
                ))
            else:
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    connectgaps=False
                ))

            if tech_indicators == "SMA20":
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA20'],
                    mode='lines',
                    name='SMA20',
                    line=dict(color='orange'),
                    connectgaps=False
                ))
            elif tech_indicators == "SMA50":
                if 'SMA50' in df.columns and not df['SMA50'].isna().all():
                    fig_price.add_trace(go.Scatter(
                        x=df.index,
                        y=df['SMA50'],
                        mode='lines',
                        name='SMA50',
                        line=dict(color='purple'),
                        connectgaps=False
                    ))
                else:
                    st.warning("SMA50 data is not available for the selected date range.")
            elif tech_indicators == "EMA20":
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['EMA20'],
                    mode='lines',
                    name='EMA20',
                    line=dict(color='green'),
                    connectgaps=False
                ))
            elif tech_indicators == "EMA50":
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['EMA50'],
                    mode='lines',
                    name='EMA50',
                    line=dict(color='red'),
                    connectgaps=False
                ))
            elif tech_indicators == "RSI":
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    connectgaps=False
                ))
                fig_rsi.update_layout(title=f"RSI for {ticker_symbol}", xaxis_title="Date", yaxis_title="RSI")
                st.plotly_chart(fig_rsi)
            elif tech_indicators == "MACD":
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    mode='lines',
                    name='MACD',
                    connectgaps=False
                ))
                fig_macd.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='orange'),
                    connectgaps=False
                ))
                fig_macd.update_layout(title=f"MACD for {ticker_symbol}", xaxis_title="Date", yaxis_title="MACD")
                st.plotly_chart(fig_macd)
            elif tech_indicators == "Bollinger Bands":
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Bollinger_Upper'],
                    mode='lines',
                    name='Bollinger Upper',
                    line=dict(color='red'),
                    connectgaps=False
                ))
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Bollinger_Lower'],
                    mode='lines',
                    name='Bollinger Lower',
                    line=dict(color='green'),
                    connectgaps=False
                ))

            if tech_indicators not in ["RSI", "MACD"]:
                fig_price.update_layout(title=f"{ticker_symbol} Price ({data_freq})", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_price)

        st.write("#### Volume Chart")
        if df['Volume'].isna().all():
            st.error("No valid volume data to plot the chart. Please check the data.")
        else:
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            fig_volume.update_layout(title=f"Volume for {ticker_symbol}", xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig_volume)

    with tab2:
        st.write("### News")
        articles = fetch_news(ticker_symbol)
        if articles:
            for article in articles:
                title = article.get("title", "No title")
                description = article.get("description", "No description")
                url = article.get("url", "#")
                published_at = article.get("publishedAt", "Unknown")
                st.write(f"**{title}**")
                st.write(f"Published at: {published_at}")
                st.write(description)
                st.write(f"[Read more]({url})")
                st.markdown("---")
        else:
            st.write("No news found for this ticker. Please check the API key or try another ticker.")

    with tab3:
        st.write("### Sentiment Analysis")
        articles = fetch_news(ticker_symbol)
        if articles:
            sentiment_data = []
            for article in articles:
                title = article.get("title", "No title")
                description = article.get("description", "No description") or ""
                content = title + " " + description
                sentiment, score = analyze_sentiment(content)
                sentiment_data.append({
                    "Title": title,
                    "Sentiment": sentiment,
                    "Confidence Score": score
                })
            
            sentiment_df = pd.DataFrame(sentiment_data)
            st.write("#### Sentiment Analysis Results")
            st.dataframe(sentiment_df)

            sentiment_counts = sentiment_df["Sentiment"].value_counts()
            st.write("#### Sentiment Distribution")
            fig = go.Figure(data=[
                go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values)
            ])
            st.plotly_chart(fig)
        else:
            st.write("No news found for sentiment analysis. Please check the API key or try another ticker.")

    with tab4:
        st.write("### Stock Price Prediction")

        # Select Data Source
        st.write("#### Select Data Source")
        data_source = st.radio("Data Source", ["Use Pricing Data", "Upload Custom Data"])
        
        if data_source == "Use Pricing Data":
            data = df.copy()
        else:
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                data['% Change'] = data['Close'].pct_change() * 100
                data['RSI'] = compute_rsi(data['Close'])
                data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])
                data['Bollinger_Upper'], data['Bollinger_Lower'] = compute_bollinger_bands(data['Close'])
                data['SMA20'] = data['Close'].rolling(window=20).mean()
                if len(data) >= 50:
                    data['SMA50'] = data['Close'].rolling(window=50).mean()
                else:
                    data['SMA50'] = np.nan
                data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
                data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
                data = data.dropna()
                if data.empty:
                    st.error("Uploaded data is empty after processing. Please upload a valid CSV file with sufficient data.")
                    st.stop()
            else:
                st.warning("Please upload a CSV file to proceed.")
                st.stop()

        # Hiển thị dữ liệu
        st.write("#### Data")
        display_df = data[['Close', 'High', 'Low', 'Open', 'Volume', '% Change', 'SMA20', 'SMA50']].reset_index()
        display_df = display_df.rename(columns={'index': 'Date'})
        display_df['% Change'] = display_df['% Change'].apply(lambda x: f"{x:.3f}%" if not pd.isna(x) else "N/A")
        st.dataframe(display_df)

        # Select Input Features
        st.write("#### Select Input Features")
        all_features = ["High", "Low", "Open", "Volume", "RSI", "MACD", "MACD_Signal", "Bollinger_Upper", "Bollinger_Lower", "SMA20", "SMA50", "EMA20", "EMA50", "% Change"]
        selected_features = st.multiselect("Features", all_features, default=["High", "Low"])
        y_feature = "Close"

        if not selected_features:
            st.warning("Please select at least one feature to proceed!")
            st.stop()

        # Select Data for Model
        st.write("#### Select Data for Model")
        model_data = data[[y_feature] + selected_features].copy()
        display_model_data = model_data.reset_index()
        display_model_data = display_model_data.rename(columns={'index': 'Date'})
        st.dataframe(display_model_data)

        # Select Train-Test Ratio
        st.write("#### Select Train-Test Ratio (%)")
        train_ratio = st.slider("Train Ratio", 50, 95, 80, 1)
        train_size = int(len(model_data) * train_ratio / 100)
        test_size = len(model_data) - train_size
        st.write(f"Train: {train_size} samples, Test: {test_size} samples")

        # Select Model
        st.write("#### Select Model")
        selected_model = st.radio("Model", ["LSTM", "Transformer", "XGBoost", "ARIMA"])

        # Number of Epochs (chỉ áp dụng cho LSTM và Transformer)
        if selected_model in ["LSTM", "Transformer"]:
            st.write("#### Number of Epochs")
            epochs = st.slider("Epochs", 1, 50, 5, 1)
        else:
            epochs = None

        def train_and_evaluate(model_name, df, selected_features, y_feature, train_size, epochs):
            df = df.dropna()
            
            if len(df) < 30:
                raise ValueError(f"Not enough data to train! At least 30 data points are required, but only {len(df)} are available.")
            
            # Chia dữ liệu thành train và test
            train_data = df.iloc[:train_size]
            test_data = df.iloc[train_size:]
            train_dates = train_data.index
            test_dates = test_data.index
            
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[[y_feature] + selected_features])
            
            time_range = 30  # Số bước thời gian cho LSTM, Transformer, XGBoost
            model = None
            
            # Dự đoán cho toàn bộ dữ liệu (train + test)
            all_dates = df.index[time_range:]  # Bỏ qua time_range đầu tiên vì không có dự đoán
            all_predictions = []
            y_true = df[y_feature].iloc[time_range:].values
            
            if model_name == "LSTM":
                X_train = []
                y_train = []
                X_all = []
                
                # Chuẩn bị dữ liệu cho train
                for i in range(len(train_data) - time_range):
                    X_train.append(scaled_data[i:i + time_range, 1:])
                    y_train.append(scaled_data[i + time_range, 0])
                
                # Chuẩn bị dữ liệu cho toàn bộ tập dữ liệu
                for i in range(len(df) - time_range):
                    X_all.append(scaled_data[i:i + time_range, 1:])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_all = np.array(X_all)
                
                if len(X_train) == 0 or X_train.shape[1] != time_range or X_train.shape[2] != len(selected_features):
                    raise ValueError(f"Invalid training data shape: {X_train.shape}, expected (samples, {time_range}, {len(selected_features)})")
                
                # Xây dựng mô hình LSTM với Attention
                inputs = Input(shape=(time_range, len(selected_features)))
                lstm1 = LSTM(100, return_sequences=True)(inputs)
                dropout1 = Dropout(0.2)(lstm1)
                lstm2 = LSTM(50, return_sequences=True)(dropout1)
                dropout2 = Dropout(0.2)(lstm2)
                lstm3 = LSTM(50, return_sequences=True)(dropout2)
                context, attention_weights = AttentionLayer()(lstm3)
                dense1 = Dense(25, activation='relu')(context)
                outputs = Dense(1)(dense1)
                model = Model(inputs=inputs, outputs=[outputs, attention_weights])
                
                model.compile(optimizer='adam', loss=['mse', None])
                model.fit(X_train, [y_train, np.zeros((len(y_train), time_range))], epochs=epochs, batch_size=32, verbose=0)
                
                # Dự đoán trên toàn bộ dữ liệu
                all_predictions = []
                all_attention_weights = []
                for i in range(len(X_all)):
                    current_input = X_all[i].reshape(1, time_range, len(selected_features))
                    pred, attn = model.predict(current_input, verbose=0)
                    all_predictions.append(pred[0, 0])
                    all_attention_weights.append(attn[0, :, 0])
                
                dummy_features = np.zeros((len(all_predictions), len(selected_features)))
                combined_predictions = np.column_stack([all_predictions, dummy_features])
                inverse_transformed = scaler.inverse_transform(combined_predictions)
                all_predictions = inverse_transformed[:, 0]
                y_true_transformed = scaler.inverse_transform(np.column_stack([scaled_data[time_range:, 0], dummy_features]))[:, 0]
            
            elif model_name == "Transformer":
                X_train = []
                y_train = []
                X_all = []
                
                for i in range(len(train_data) - time_range):
                    X_train.append(scaled_data[i:i + time_range, 1:])
                    y_train.append(scaled_data[i + time_range, 0])
                
                for i in range(len(df) - time_range):
                    X_all.append(scaled_data[i:i + time_range, 1:])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_all = np.array(X_all)
                
                if len(X_train) == 0 or X_train.shape[1] != time_range or X_train.shape[2] != len(selected_features):
                    raise ValueError(f"Invalid training data shape: {X_train.shape}, expected (samples, {time_range}, {len(selected_features)})")
                
                # Xây dựng mô hình Transformer mà không lấy trọng số attention
                inputs = Input(shape=(time_range, len(selected_features)))
                attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs, inputs)  # Bỏ return_attention_scores
                x = LayerNormalization(epsilon=1e-6)(attn_output + inputs)
                x = LSTM(50, return_sequences=False)(x)
                x = Dense(25, activation='relu')(x)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)  # Chỉ có một đầu ra
                
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
                
                all_predictions = []
                all_attention_weights = None  # Không có trọng số attention
                for i in range(len(X_all)):
                    current_input = X_all[i].reshape(1, time_range, len(selected_features))
                    pred = model.predict(current_input, verbose=0)
                    all_predictions.append(pred[0, 0])
                
                dummy_features = np.zeros((len(all_predictions), len(selected_features)))
                combined_predictions = np.column_stack([all_predictions, dummy_features])
                inverse_transformed = scaler.inverse_transform(combined_predictions)
                all_predictions = inverse_transformed[:, 0]
                y_true_transformed = scaler.inverse_transform(np.column_stack([scaled_data[time_range:, 0], dummy_features]))[:, 0]
            
            elif model_name == "XGBoost":
                X_train = []
                y_train = []
                X_all = []
                
                for i in range(len(train_data) - time_range):
                    X_train.append(scaled_data[i:i + time_range, 1:].flatten())
                    y_train.append(scaled_data[i + time_range, 0])
                
                for i in range(len(df) - time_range):
                    X_all.append(scaled_data[i:i + time_range, 1:].flatten())
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_all = np.array(X_all)
                
                model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                
                all_predictions = model.predict(X_all)
                dummy_features = np.zeros((len(all_predictions), len(selected_features)))
                combined_predictions = np.column_stack([all_predictions, dummy_features])
                inverse_transformed = scaler.inverse_transform(combined_predictions)
                all_predictions = inverse_transformed[:, 0]
                y_true_transformed = scaler.inverse_transform(np.column_stack([scaled_data[time_range:, 0], dummy_features]))[:, 0]
                all_attention_weights = None
            
            elif model_name == "ARIMA":
                train_series = train_data[y_feature]
                model = ARIMA(train_series, order=(5, 1, 0))
                model_fit = model.fit()
                
                all_predictions = model_fit.predict(start=time_range, end=len(df) - 1, typ='levels')
                all_predictions = all_predictions.values
                y_true_transformed = y_true
                scaler = None
                all_attention_weights = None
            
            return all_dates, all_predictions, y_true_transformed, model, scaler, all_attention_weights

        def predict_future(model_name, df, selected_features, y_feature, model, scaler, forecast_horizon, confidence_level=0.95):
            df = df.dropna()
            time_range = 30
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')
            
            if model_name == "ARIMA":
                model_fit = model
                forecast = model_fit.forecast(steps=forecast_horizon)
                predictions = forecast.values
                forecast_obj = model_fit.get_forecast(steps=forecast_horizon)
                conf_int = forecast_obj.conf_int(alpha=1 - confidence_level)
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            else:
                scaled_data = scaler.transform(df[[y_feature] + selected_features])
                
                if model_name in ["LSTM", "Transformer"]:
                    last_sequence = scaled_data[-time_range:, 1:]
                    predictions = []
                    temp_sequence = last_sequence.copy()
                    
                    all_predictions = []
                    for _ in range(50):
                        temp_sequence = last_sequence.copy()
                        preds = []
                        for _ in range(forecast_horizon):
                            current_input = temp_sequence.reshape(1, time_range, len(selected_features))
                            if model_name == "LSTM":
                                next_y_pred, _ = model.predict(current_input, verbose=0)
                            else:  # Transformer
                                next_y_pred = model.predict(current_input, verbose=0)
                            next_y_pred = next_y_pred[0, 0]
                            preds.append(next_y_pred)
                            temp_sequence = np.roll(temp_sequence, -1, axis=0)
                            temp_sequence[-1, 0] = next_y_pred + np.random.normal(0, 0.01)
                        all_predictions.append(preds)
                    
                    all_predictions = np.array(all_predictions)
                    predictions = np.mean(all_predictions, axis=0)
                    std_predictions = np.std(all_predictions, axis=0)
                    z_score = 1.96 if confidence_level == 0.95 else 2.58
                    lower_bound = predictions - z_score * std_predictions
                    upper_bound = predictions + z_score * std_predictions
                    
                    dummy_features = np.zeros((forecast_horizon, len(selected_features)))
                    combined_predictions = np.column_stack([predictions, dummy_features])
                    combined_lower = np.column_stack([lower_bound, dummy_features])
                    combined_upper = np.column_stack([upper_bound, dummy_features])
                    
                    predictions = scaler.inverse_transform(combined_predictions)[:, 0]
                    lower_bound = scaler.inverse_transform(combined_lower)[:, 0]
                    upper_bound = scaler.inverse_transform(combined_upper)[:, 0]
                
                else:  # XGBoost
                    last_sequence = scaled_data[-time_range:, 1:].flatten()
                    predictions = []
                    temp_sequence = last_sequence.copy()
                    
                    all_predictions = []
                    for _ in range(50):
                        temp_sequence = last_sequence.copy()
                        preds = []
                        for _ in range(forecast_horizon):
                            current_input = temp_sequence.reshape(1, -1)
                            next_y_pred = model.predict(current_input)[0]
                            preds.append(next_y_pred)
                            temp_sequence = np.roll(temp_sequence, -len(selected_features))
                            temp_sequence[-len(selected_features):] = next_y_pred + np.random.normal(0, 0.01)
                        all_predictions.append(preds)
                    
                    all_predictions = np.array(all_predictions)
                    predictions = np.mean(all_predictions, axis=0)
                    std_predictions = np.std(all_predictions, axis=0)
                    z_score = 1.96 if confidence_level == 0.95 else 2.58
                    lower_bound = predictions - z_score * std_predictions
                    upper_bound = predictions + z_score * std_predictions
                    
                    dummy_features = np.zeros((forecast_horizon, len(selected_features)))
                    combined_predictions = np.column_stack([predictions, dummy_features])
                    combined_lower = np.column_stack([lower_bound, dummy_features])
                    combined_upper = np.column_stack([upper_bound, dummy_features])
                    
                    predictions = scaler.inverse_transform(combined_predictions)[:, 0]
                    lower_bound = scaler.inverse_transform(combined_lower)[:, 0]
                    upper_bound = scaler.inverse_transform(combined_upper)[:, 0]
            
            return future_dates, predictions, lower_bound, upper_bound

        if st.button("Train and Evaluate"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Training {selected_model} model...")
            try:
                all_dates, all_predictions, y_true, model, scaler, all_attention_weights = train_and_evaluate(
                    selected_model, model_data, selected_features, y_feature, train_size, epochs
                )
                st.session_state['all_dates'] = all_dates
                st.session_state['all_predictions'] = all_predictions
                st.session_state['y_true'] = y_true
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['all_attention_weights'] = all_attention_weights
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                mae, mse, rmse, r2, mape, smape = calculate_metrics(y_true, all_predictions)
                st.session_state['metrics'] = {
                    'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mape': mape, 'smape': smape
                }
                
                # Biểu đồ dự đoán trên toàn bộ dữ liệu
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=model_data.index,
                    y=model_data[y_feature],
                    mode='lines',
                    name="Actual Price",
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=all_dates,
                    y=all_predictions,
                    mode='lines',
                    name=f"{selected_model} Prediction",
                    line=dict(color='orange', dash='dash')
                ))
                fig.update_layout(title=f"Prediction for {selected_model}", xaxis_title="Date", yaxis_title="Price")
                st.session_state['prediction_chart'] = fig
            
            except Exception as e:
                st.error(f"Error training {selected_model} model: {str(e)}")
                status_text.text("Training failed!")

        # Hiển thị lại kết quả nếu đã có trong session_state
        if 'metrics' in st.session_state:
            st.write("#### Model Evaluation")
            metrics = st.session_state['metrics']
            st.write(f"MAE: {metrics['mae']:.2f}")
            st.write(f"MSE: {metrics['mse']:.2f}")
            st.write(f"RMSE: {metrics['rmse']:.2f}")
            st.write(f"R²: {metrics['r2']:.2f}")
            st.write(f"MAPE: {metrics['mape']:.2f}%")
            st.write(f"SMAPE: {metrics['smape']:.2f}%")

        if 'prediction_chart' in st.session_state:
            st.write("#### Prediction Chart")
            st.write(f"Prediction for {selected_model}")
            st.plotly_chart(st.session_state['prediction_chart'])

        # Multi-step Future Prediction
        st.write("#### Multi-step Future Prediction")
        forecast_horizon = st.slider("Number of sessions to predict", 1, 30, 5, 1)
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95, 1) / 100
        
        if st.button("Predict Future Sessions"):
            if 'model' not in st.session_state:
                st.error("Please train the model first!")
            else:
                model = st.session_state['model']
                scaler = st.session_state['scaler']
                
                future_dates, future_predictions, lower_bound, upper_bound = predict_future(
                    selected_model, model_data, selected_features, y_feature, model, scaler, forecast_horizon, confidence_level
                )
                
                # Lưu kết quả vào session_state
                st.session_state['future_dates'] = future_dates
                st.session_state['future_predictions'] = future_predictions
                st.session_state['lower_bound'] = lower_bound
                st.session_state['upper_bound'] = upper_bound
                
                # Hiển thị giá dự đoán cho phiên tiếp theo
                st.session_state['next_session_price'] = future_predictions[0]
                
                # Biểu đồ khoảng tin cậy
                fig_interval = go.Figure()
                fig_interval.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines',
                    name="Prediction",
                    line=dict(color='blue')
                ))
                fig_interval.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    mode='lines',
                    name="Upper Bound",
                    line=dict(color='red', dash='dash')
                ))
                fig_interval.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    mode='lines',
                    name="Lower Bound",
                    line=dict(color='green', dash='dash')
                ))
                fig_interval.update_layout(title="Prediction Intervals", xaxis_title="Date", yaxis_title="Price")
                st.session_state['interval_chart'] = fig_interval
                
                # Biểu đồ dự đoán tương lai
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=model_data.index[-60:],
                    y=model_data[y_feature].iloc[-60:],
                    mode='lines',
                    name="Actual Price",
                    line=dict(color='blue')
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines',
                    name=f"{selected_model} Prediction",
                    line=dict(color='orange', dash='dash')
                ))
                fig_future.update_layout(title=f"Future Prediction for {selected_model}", xaxis_title="Date", yaxis_title="Price")
                st.session_state['future_chart'] = fig_future

        # Hiển thị lại kết quả nếu đã có trong session_state
        if 'next_session_price' in st.session_state:
            st.write(f"#### Predicted Price for Next Session: {st.session_state['next_session_price']:.2f}")

        if 'lower_bound' in st.session_state and 'upper_bound' in st.session_state:
            st.write("#### Prediction Intervals")
            st.write(f"Confidence Level: {int(confidence_level * 100)}%")
            st.write(f"Interval (Confidence Level {int(confidence_level * 100)}%): [{st.session_state['lower_bound'][0]:.2f}, {st.session_state['upper_bound'][0]:.2f}]")

        if 'interval_chart' in st.session_state:
            st.plotly_chart(st.session_state['interval_chart'])

        if 'future_chart' in st.session_state:
            st.write("#### Future Prediction Chart")
            st.plotly_chart(st.session_state['future_chart'])

        # Explain Prediction
        st.write("#### Explain Prediction")
        explain_method = st.selectbox("Choose method", ["SHAP", "LIME", "Attention"])
        
        if st.button("Explain"):
            if 'model' not in st.session_state:
                st.error("Please train the model first!")
            else:
                model = st.session_state['model']
                scaler = st.session_state['scaler']
                all_attention_weights = st.session_state.get('all_attention_weights', None)
                train_data = model_data.iloc[:train_size]
                
                if explain_method == "SHAP":
                    st.write(f"#### SHAP - Feature Importance ({selected_model})")
                    
                    try:
                        scaled_data = scaler.transform(model_data[[y_feature] + selected_features])
                        X_full = scaled_data[:train_size, 1:]
                        time_range = 30
                        
                        n_background_samples = min(50, len(X_full) - time_range + 1)
                        if n_background_samples <= 0:
                            raise ValueError("Not enough data to create background samples for SHAP.")
                        
                        X_background = []
                        for i in range(n_background_samples):
                            X_background.append(X_full[i:i + time_range, :].flatten())
                        X_background = np.array(X_background)
                        
                        X_explain = X_full[-time_range:, :].flatten().reshape(1, -1)
                        
                        if selected_model in ["LSTM", "Transformer"]:
                            def model_predict(x):
                                x_3d = x.reshape(-1, time_range, len(selected_features))
                                if selected_model == "LSTM":
                                    pred, _ = model.predict(x_3d, verbose=0)
                                else:  # Transformer
                                    pred = model.predict(x_3d, verbose=0)
                                return pred.reshape(-1,)
                        else:
                            model_predict = model.predict
                        
                        explainer = shap.KernelExplainer(model_predict, X_background)
                        shap_values = explainer.shap_values(X_explain)
                        
                        expanded_feature_names = []
                        for t in range(time_range):
                            for feat in selected_features:
                                expanded_feature_names.append(f"{feat} (t-{time_range-t-1})")
                        
                        plt.figure(figsize=(12, 8))
                        shap.summary_plot(
                            shap_values,
                            X_explain,
                            feature_names=expanded_feature_names,
                            plot_type="bar",
                            show=False
                        )
                        st.pyplot(plt.gcf())
                        plt.clf()
                    
                    except Exception as e:
                        st.error(f"Error generating SHAP for {selected_model}: {str(e)}")
                
                elif explain_method == "LIME":
                    st.write(f"#### LIME - Feature Importance ({selected_model})")
                    
                    try:
                        scaled_data = scaler.transform(model_data[[y_feature] + selected_features])
                        X_full = scaled_data[:train_size, 1:]
                        time_range = 30
                        
                        X_lime_train = []
                        for i in range(len(X_full) - time_range + 1):
                            X_lime_train.append(X_full[i:i + time_range, :].flatten())
                        X_lime_train = np.array(X_lime_train)
                        
                        if len(X_lime_train) == 0:
                            raise ValueError("Not enough data to create training samples for LIME.")
                        
                        X_explain = X_full[-time_range:, :].flatten()
                        
                        if selected_model in ["LSTM", "Transformer"]:
                            def model_predict(x):
                                n_samples = x.shape[0]
                                x_3d = x.reshape(n_samples, time_range, len(selected_features))
                                if selected_model == "LSTM":
                                    pred, _ = model.predict(x_3d, verbose=0)
                                else:  # Transformer
                                    pred = model.predict(x_3d, verbose=0)
                                return pred.flatten()
                        else:
                            model_predict = model.predict
                        
                        expanded_feature_names = []
                        for t in range(time_range):
                            for feat in selected_features:
                                expanded_feature_names.append(f"{feat} (t-{time_range-t-1})")
                        
                        explainer = lime.lime_tabular.LimeTabularExplainer(
                            training_data=X_lime_train,
                            feature_names=expanded_feature_names,
                            mode="regression"
                        )
                        
                        exp = explainer.explain_instance(
                            data_row=X_explain,
                            predict_fn=model_predict,
                            num_features=5
                        )
                        
                        fig = exp.as_pyplot_figure()
                        st.pyplot(fig)
                        plt.clf()
                    
                    except Exception as e:
                        st.error(f"Error generating LIME for {selected_model}: {str(e)}")
                
                elif explain_method == "Attention":
                    st.write(f"#### Attention - Timestep Importance ({selected_model})")
                    
                    if selected_model != "LSTM":
                        st.error("Attention mechanism is only available for LSTM models. Transformer attention has been disabled.")
                    elif all_attention_weights is None:
                        st.error("Attention weights are not available. Please retrain the model.")
                    else:
                        try:
                            time_range = 30
                            # Lấy trọng số Attention của mẫu cuối cùng
                            attention_weights = all_attention_weights[-1]  # Shape: (time_range,)
                            plt.figure(figsize=(10, 6))
                            plt.imshow(attention_weights.reshape(1, -1), cmap='viridis', aspect='auto')
                            plt.colorbar(label='Attention Weight')
                            plt.title("Attention Weights for LSTM")
                            plt.xlabel("Timestep")
                            plt.ylabel("Attention")
                            plt.xticks(np.arange(time_range), [f"t-{time_range-i-1}" for i in range(time_range)], rotation=45)
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.clf()
                        except Exception as e:
                            st.error(f"Error generating Attention for {selected_model}: {str(e)}")

else:
    st.error("Unable to load data. Please check the ticker, date range, or network connection!")