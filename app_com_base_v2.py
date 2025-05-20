
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
    df['Total jogo anterior'] = df['Total 1P Anterior'] + df['Total 2P Anterior']
    df = df.dropna()
    df['Target'] = (df['Total 2ª parte'] > 4.5).astype(int)
    X = df[['Total 1ª parte','Total 1P Anterior','Total 2P Anterior','Total jogo anterior']]
    y = df['Target']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(Xs, y)
    return model, scaler, df

st.title("App com Base de Dados - Previsão Over 4.5 (2ª Parte)")

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
    total_1p = st.number_input("Total 1ª Parte (atual)", step=1, value=5)

    ult_1p = int(df_base['Total 1ª parte'].iloc[-1])
    ult_2p = int(df_base['Total 2ª parte'].iloc[-1])
    ult_total = ult_1p + ult_2p

    if st.button("Prever 2ª Parte"):
        X_new = np.array([[total_1p, ult_1p, ult_2p, ult_total]])
        Xs_new = scaler.transform(X_new)
        pred = model.predict(Xs_new)[0]
        prob = model.predict_proba(Xs_new)[0][1]
        if pred == 1:
            st.markdown(f"<span style='color:darkgreen'><strong>OVER 4.5 (2ª parte)</strong> – {prob*100:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'><strong>UNDER 4.5 (2ª parte)</strong> – {(1-prob)*100:.1f}%</span>", unsafe_allow_html=True)

    st.download_button(
        label="📥 Baixar base com colunas calculadas",
        data=df_trained.to_csv(index=False),
        file_name="base_atualizada.csv",
        mime="text/csv"
    )
