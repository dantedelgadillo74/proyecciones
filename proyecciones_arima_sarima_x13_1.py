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

# Configuracion de la pagina
st.set_page_config(layout="wide")
st.title("Proyecciones de Asegurados IMSS - Jalisco")

# Cargar archivo Excel
archivo_excel = st.file_uploader("Carga el archivo de asegurados históricos", type=["xlsx"])

if archivo_excel:
    df = pd.read_excel(archivo_excel, sheet_name="ta_total")
    df["fecha"] = pd.to_datetime(df["Mes"])
    df.set_index("fecha", inplace=True)
    serie = df["ta_total_jalisco"]

    # Proyeccion SARIMA
    modelo_sarima = SARIMAX(serie, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    resultado_sarima = modelo_sarima.fit(disp=False)
    forecast_sarima = resultado_sarima.get_forecast(steps=24)
    forecast_mean = forecast_sarima.predicted_mean
    forecast_ci = forecast_sarima.conf_int()

    # Proyeccion ARIMA (no estacional)
    modelo_arima = sm.tsa.ARIMA(serie, order=(1, 1, 1))
    resultado_arima = modelo_arima.fit()
    forecast_arima = resultado_arima.get_forecast(steps=24)
    forecast_arima_mean = forecast_arima.predicted_mean
    forecast_arima_ci = forecast_arima.conf_int()

    # Fallback: regresion lineal como proxy de X13
    future_index = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1), periods=24, freq="MS")
    x = np.arange(len(serie))
    y = serie.values
    coef = np.polyfit(x, y, deg=1)
    x_future = np.arange(len(serie), len(serie) + 24)
    forecast_x13 = pd.Series(np.polyval(coef, x_future), index=future_index)

    # Consolidar proyecciones en resumen
    df_resumen = pd.DataFrame({
        "fecha": future_index,
        "proyeccion_sarima": forecast_mean.values,
        "proyeccion_x13": forecast_x13.values,
        "proyeccion_arima": forecast_arima_mean.values
    })
    df_resumen["Año"] = df_resumen["fecha"].dt.year
    df_resumen["Mes"] = df_resumen["fecha"].dt.month
    df_resumen["Nombre_mes"] = df_resumen["fecha"].dt.strftime('%B')

    # Porcentaje de diferencia entre modelos
    df_resumen["%_ARIMA_vs_SARIMA"] = ((df_resumen["proyeccion_arima"] - df_resumen["proyeccion_sarima"]) / df_resumen["proyeccion_sarima"]) * 100
    df_resumen["%_X13_vs_SARIMA"] = ((df_resumen["proyeccion_x13"] - df_resumen["proyeccion_sarima"]) / df_resumen["proyeccion_sarima"]) * 100

    # Filtro de años
    min_year = df_resumen["Año"].min()
    max_year = df_resumen["Año"].max()
    st.sidebar.header("Filtros")
    rango_anios = st.sidebar.slider("Selecciona el rango de años", min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)
    df_resumen_filtrado = df_resumen[(df_resumen["Año"] >= rango_anios[0]) & (df_resumen["Año"] <= rango_anios[1])]
    future_index_filtrado = df_resumen_filtrado["fecha"]

    # Filtro de modelos a mostrar
    modelos = st.sidebar.multiselect("Selecciona modelos a mostrar", ["Histórico", "SARIMA", "ARIMA", "X13/Fallback"], default=["Histórico", "SARIMA", "ARIMA"])

 # Fechas válidas presentes en forecast
    fechas_sarima_validas = forecast_mean.index.intersection(future_index_filtrado)
    fechas_arima_validas = forecast_arima_mean.index.intersection(future_index_filtrado)
    fechas_x13_validas = forecast_x13.index.intersection(future_index_filtrado)

    # Gráfica
    fig, ax = plt.subplots(figsize=(14, 6))

    if "Histórico" in modelos:
        ax.plot(serie, label="Histórico", color="black")

    if "SARIMA" in modelos and not fechas_sarima_validas.empty:
        ax.plot(forecast_mean.loc[fechas_sarima_validas], label="Proyección SARIMA", color="blue")
        ax.fill_between(forecast_ci.loc[fechas_sarima_validas].index,
                        forecast_ci.loc[fechas_sarima_validas].iloc[:, 0],
                        forecast_ci.loc[fechas_sarima_validas].iloc[:, 1],
                        color="blue", alpha=0.2)

    if "ARIMA" in modelos and not fechas_arima_validas.empty:
        ax.plot(forecast_arima_mean.loc[fechas_arima_validas], label="Proyección ARIMA", color="orange")
        ax.fill_between(forecast_arima_ci.loc[fechas_arima_validas].index,
                        forecast_arima_ci.loc[fechas_arima_validas].iloc[:, 0],
                        forecast_arima_ci.loc[fechas_arima_validas].iloc[:, 1],
                        color="orange", alpha=0.2)

    if "X13/Fallback" in modelos and not fechas_x13_validas.empty:
        ax.plot(forecast_x13.loc[fechas_x13_validas], label="Proyección X13", color="green")

    ax.legend()
    ax.grid(True)
    ax.set_title(f"Proyecciones {rango_anios[0]} - {rango_anios[1]}")
    st.pyplot(fig)


    # Mostrar tabla resumen
    st.subheader("Resumen de proyecciones por mes")
    st.dataframe(df_resumen_filtrado.style.format({
        "proyeccion_sarima": "{:,.0f}",
        "proyeccion_x13": "{:,.0f}",
        "proyeccion_arima": "{:,.0f}",
        "%_ARIMA_vs_SARIMA": "{:.2f}%",
        "%_X13_vs_SARIMA": "{:.2f}%"
    }))

else:
    st.info("Por favor carga un archivo Excel con la hoja 'Historico' que contenga las columnas 'fecha' y 'ta_total_jalisco'.")
