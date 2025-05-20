
import streamlit as st
import joblib
import numpy as np

modelo = joblib.load("modelo_over2p.joblib")
scaler = joblib.load("escalador_over2p.joblib")

st.title("Previsão: Over 4.5 na 2ª Parte")

st.write("Previsão baseada no jogo atual + jogo anterior:")

col1, col2 = st.columns(2)
with col1:
    total_1p = st.number_input("Total 1ª parte (jogo atual)", value=5)
    total_jogo = st.number_input("Total do jogo (jogo atual)", value=13)
with col2:
    total_1p_ant = st.number_input("Total 1ª parte (jogo anterior)", value=6)
    total_2p_ant = st.number_input("Total 2ª parte (jogo anterior)", value=5)
    total_jogo_ant = st.number_input("Total do jogo (jogo anterior)", value=13)

if st.button("Prever"):
    entrada = np.array([[total_1p, total_jogo, total_1p_ant, total_2p_ant, total_jogo_ant]])
    entrada_scaled = scaler.transform(entrada)
    predicao = modelo.predict(entrada_scaled)[0]
    prob = modelo.predict_proba(entrada_scaled)[0][1]

    if predicao == 1:
        st.success(f"🔵 **Previsão: OVER 4.5 na 2ª parte** ({prob*100:.1f}% confiança)")
    else:
        st.error(f"🔴 **Previsão: UNDER 4.5 na 2ª parte** ({(1 - prob)*100:.1f}% confiança)")
