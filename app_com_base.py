
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

@st.cache_data
def train_model(df):
    df = df.copy()
    df['Total 1P Anterior'] = df['Total 1ª parte'].shift(1)
    df['Total 2P Anterior'] = df['Total 2ª parte'].shift(1)
    df['Total jogo anterior'] = df['Total jogo'].shift(1)
    df = df.dropna()
    df['Target'] = (df['Total 2ª parte'] > 4.5).astype(int)
    X = df[['Total 1ª parte','Total jogo','Total 1P Anterior','Total 2P Anterior','Total jogo anterior']]
    y = df['Target']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(Xs, y)
    return model, scaler, df

st.title("App com Base de Dados - Over 4.5 na 2ª Parte")

uploaded = st.file_uploader("Upload da base de dados (.xlsx ou .csv)", type=["xlsx","csv"])
if uploaded:
    if uploaded.name.endswith('.xlsx'):
        df_base = pd.read_excel(uploaded)
    else:
        df_base = pd.read_csv(uploaded)
    st.subheader("Base de Dados Atual")
    st.dataframe(df_base)
    model, scaler, df_trained = train_model(df_base)
    st.success("Modelo treinado com sucesso!")

    st.subheader("Previsão do Próximo Jogo")
    cols = st.columns(2)
    with cols[0]:
        total_1p = st.number_input("Total 1ª Parte (atual)", step=1, value=5)
        total_jogo = st.number_input("Total do Jogo (atual)", step=1, value=10)
    with cols[1]:
        total_1p_ant = st.number_input("Total 1P (anterior)", step=1, value=int(df_base['Total 1ª parte'].iloc[-1]))
        total_2p_ant = st.number_input("Total 2P (anterior)", step=1, value=int(df_base['Total 2ª parte'].iloc[-1]))
        total_jogo_ant = st.number_input("Total do Jogo (anterior)", step=1, value=int(df_base['Total jogo'].iloc[-1]))

    if st.button("Prever 2ª Parte"):
        X_new = np.array([[total_1p, total_jogo, total_1p_ant, total_2p_ant, total_jogo_ant]])
        Xs_new = scaler.transform(X_new)
        pred = model.predict(Xs_new)[0]
        prob = model.predict_proba(Xs_new)[0][1]
        if pred == 1:
            st.markdown(f"<span style='color:darkgreen'><strong>OVER 4.5 (2ª parte)</strong> – {prob*100:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'><strong>UNDER 4.5 (2ª parte)</strong> – {(1-prob)*100:.1f}%</span>", unsafe_allow_html=True)

    st.download_button(
        label="📥 Baixar base atual com colunas calculadas",
        data=df_trained.to_csv(index=False),
        file_name="base_atualizada.csv",
        mime="text/csv"
    )
