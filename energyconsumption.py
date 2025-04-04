import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
import plotly.graph_objs as go

st.set_page_config(page_title="Turkey Energy Forecast", layout="wide")

st.title("🔋 Predicting Energy Consumption of Turkey")
st.write("Upload your historical energy consumption data for forecasting using Prophet.")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with 'Date' and 'Consumption' columns", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Rename columns for Prophet
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'ds'}, inplace=True)
    if 'Consumption' in df.columns:
        df.rename(columns={'Consumption': 'y'}, inplace=True)

    df['ds'] = pd.to_datetime(df['ds'])

    st.subheader("📈 Raw Data")
    st.write(df.tail())

    # Plot original data
    st.subheader("🧾 Historical Energy Consumption")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Energy Consumption'))
    st.plotly_chart(fig, use_container_width=True)

    # Forecasting horizon
    periods = st.slider("Select number of days to forecast:", 30, 365, 90)

    # Prophet model
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Show forecasted data
    st.subheader("🔮 Forecasted Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Plot forecast
    st.subheader("📊 Forecast Plot")
    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Components
    st.subheader("🧩 Forecast Components")
    comp_fig = model.plot_components(forecast)
    st.write(comp_fig)
else:
    st.info("Please upload a CSV file to begin.")
