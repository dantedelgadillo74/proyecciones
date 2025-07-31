# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 22:36:14 2025
@author: jalis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import requests
from io import BytesIO

# Configuración de la página
st.set_page_config(layout="wide")
st.title("Proyecciones de Asegurados IMSS - Jalisco")

# --- URL raw de GitHub al archivo histórico ---
url_excel = st.sidebar.text_input(
    "URL raw del archivo Excel en GitHub",
    value="https://github.com/dantedelgadillo74/proyecciones/raw/refs/heads/main/Historico_ta_Jalisco_x_Mes.xlsx"  
)

if url_excel:
    try:
        response = requests.get(url_excel)
        response.raise_for_status()
        archivo_excel = BytesIO(response.content)

        df = pd.read_excel(archivo_excel, sheet_name="ta_total")
        df["fecha"] = pd.to_datetime(df["Mes"])
        df.set_index("fecha", inplace=True)
        serie = df["ta_total_jalisco"].asfreq("MS")

        # Proyección SARIMA
        modelo_sarima = SARIMAX(serie, order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        resultado_sarima = modelo_sarima.fit(disp=False)
        forecast_sarima = resultado_sarima.get_forecast(steps=24)
        forecast_mean = forecast_sarima.predicted_mean
        forecast_ci = forecast_sarima.conf_int()

        # Proyección ARIMA
        modelo_arima = sm.tsa.ARIMA(serie, order=(1, 1, 1))
        resultado_arima = modelo_arima.fit()
        forecast_arima = resultado_arima.get_forecast(steps=24)
        forecast_arima_mean = forecast_arima.predicted_mean
        forecast_arima_ci = forecast_arima.conf_int()

        # Proxy para X13 (regresión lineal)
        future_index = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1),
                                     periods=24, freq="MS")
        x = np.arange(len(serie))
        y = serie.values
        coef = np.polyfit(x, y, 1)
        x_future = np.arange(len(serie), len(serie) + 24)
        forecast_x13 = pd.Series(np.polyval(coef, x_future), index=future_index)

        # Consolidación de proyecciones
        df_resumen = pd.DataFrame({
            "fecha": future_index,
            "proyeccion_sarima": forecast_mean.values,
            "proyeccion_x13": forecast_x13.values,
            "proyeccion_arima": forecast_arima_mean.values
        })
        df_resumen["Año"] = df_resumen["fecha"].dt.year
        df_resumen["Mes"] = df_resumen["fecha"].dt.month
        df_resumen["Nombre_mes"] = df_resumen["fecha"].dt.strftime("%B")

        df_resumen["%_ARIMA_vs_SARIMA"] = (
            (df_resumen["proyeccion_arima"] - df_resumen["proyeccion_sarima"]) /
            df_resumen["proyeccion_sarima"]) * 100
        df_resumen["%_X13_vs_SARIMA"] = (
            (df_resumen["proyeccion_x13"] - df_resumen["proyeccion_sarima"]) /
            df_resumen["proyeccion_sarima"]) * 100

        # Panel de filtros lateral
        min_year = df_resumen["Año"].min()
        max_year = df_resumen["Año"].max()
        st.sidebar.header("Filtros")
        rango_anios = st.sidebar.slider("Rango de años", min_value=min_year,
                                         max_value=max_year,
                                         value=(min_year, max_year), step=1)

        modelos = st.sidebar.multiselect(
            "Modelos a mostrar",
            ["Histórico", "SARIMA", "ARIMA", "X13/Fallback"],
            default=["Histórico", "SARIMA", "ARIMA"]
        )

        df_filtrado = df_resumen[
            (df_resumen["Año"] >= rango_anios[0]) &
            (df_resumen["Año"] <= rango_anios[1])
        ]

        # Gráfico comparativo
        fig, ax = plt.subplots(figsize=(14, 6))
        if "Histórico" in modelos:
            ax.plot(serie, label="Histórico", color="black")
        if "SARIMA" in modelos:
            ax.plot(forecast_mean, label="SARIMA", color="blue")
            ax.fill_between(forecast_ci.index,
                            forecast_ci.iloc[:, 0],
                            forecast_ci.iloc[:, 1],
                            color="blue", alpha=0.2)
        if "ARIMA" in modelos:
            ax.plot(forecast_arima_mean, label="ARIMA", color="orange")
            ax.fill_between(forecast_arima_ci.index,
                            forecast_arima_ci.iloc[:, 0],
                            forecast_arima_ci.iloc[:, 1],
                            color="orange", alpha=0.2)
        if "X13/Fallback" in modelos:
            ax.plot(forecast_x13, label="X13 proxy", color="green")

        ax.set_title(f"Proyecciones IMSS – Jalisco ({rango_anios[0]}–{rango_anios[1]})")
        ax.set_ylabel("Asegurados")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Tabla resumen
        st.subheader("Resumen mensual de proyecciones")
        st.dataframe(
            df_filtrado.style.format({
                "proyeccion_sarima": "{:,.0f}",
                "proyeccion_x13": "{:,.0f}",
                "proyeccion_arima": "{:,.0f}",
                "%_ARIMA_vs_SARIMA": "{:.2f}%",
                "%_X13_vs_SARIMA": "{:.2f}%"
            })
        )
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo: {e}")
else:
    st.info("Introduce la URL raw del archivo Historico_ta_Jalisco_x_Mes.xlsx en GitHub.")
